#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Build a DuckDB activation database capturing top-K, bottom-K, and random-M
activations per feature across all layers. Text is stored in a separate
dataset_index table and recovered at query time.
"""

import os
import sys
import argparse
import numpy as np
import torch
import duckdb
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Add project root to path so we can import core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hooks.model_hooks import setup_hooks

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Build DuckDB activation database")
parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="Model name/path")
parser.add_argument("--top_k", type=int, default=200, help="Highest activations per feature to store")
parser.add_argument("--bottom_k", type=int, default=200, help="Lowest activations per feature to store")
parser.add_argument("--random_k", type=int, default=50, help="Random reservoir samples per feature")
parser.add_argument("--num_samples", type=int, default=1000, help="Number of dataset samples (-1 = all)")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for model inference")
parser.add_argument("--db_path", type=str, default=None, help="Output DuckDB path (default: max_activations/<model_slug>.duckdb)")
parser.add_argument("--dataset", type=str, default="monology/pile-uncopyrighted", help="HuggingFace dataset name")
parser.add_argument("--max_length", type=int, default=512, help="Max token length per sample")
parser.add_argument("--architecture", type=str, default="auto", help="Model architecture (auto, llama, gpt2)")
parser.add_argument("--capture_stage", type=str, default="default", help="Capture stage (default=gate*up, down_proj, etc.)")
args = parser.parse_args()

# --- Resolve db_path ---
model_slug = args.model.replace("/", "_")
if args.db_path is None:
    args.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{model_slug}.duckdb")

print(f"Configuration:")
print(f"  Model:          {args.model}")
print(f"  Top-K:          {args.top_k}")
print(f"  Bottom-K:       {args.bottom_k}")
print(f"  Random-K:       {args.random_k}")
print(f"  Num samples:    {'all' if args.num_samples == -1 else args.num_samples}")
print(f"  Batch size:     {args.batch_size}")
print(f"  DB path:        {args.db_path}")
print(f"  Dataset:        {args.dataset}")
print(f"  Max length:     {args.max_length}")
print(f"  Architecture:   {args.architecture}")
print(f"  Capture stage:  {args.capture_stage}")


# ---------------------------------------------------------------------------
# DuckDB schema helpers
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> duckdb.DuckDBPyConnection:
    """Create a fresh database with the required schema."""
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE dataset_index (
            data_id INTEGER PRIMARY KEY,
            text    TEXT
        )
    """)
    con.execute("""
        CREATE TABLE activations (
            layer       SMALLINT,
            feature     INTEGER,
            data_id     INTEGER,
            token_pos   SMALLINT,
            activation  FLOAT,
            bucket      TEXT
        )
    """)
    return con


def _insert_dataset_index_batch(con: duckdb.DuckDBPyConnection,
                                 data_ids: list, texts: list) -> None:
    """Bulk-insert a batch of (data_id, text) rows."""
    con.executemany("INSERT INTO dataset_index VALUES (?, ?)",
                    list(zip(data_ids, texts)))


# ---------------------------------------------------------------------------
# ActivationStore — GPU torch tensors, flushed to DB at end
# ---------------------------------------------------------------------------

class ActivationStore:
    """
    Maintains top-K, bottom-K, and reservoir-random-K activations per feature
    for every layer, using GPU torch tensors for speed.

    Memory layout (per layer):
      top_activations[layer]  : (top_k, num_features)    float32  GPU
      top_data_ids[layer]     : (top_k, num_features)    int32    GPU
      top_token_pos[layer]    : (top_k, num_features)    int32    GPU
      bot_*                   : same shape, bottom_k
      rnd_*                   : same shape, random_k
    """

    def __init__(self, num_layers: int, num_features: int,
                 top_k: int, bottom_k: int, random_k: int, device: torch.device):
        self.num_layers  = num_layers
        self.num_features = num_features
        self.top_k    = top_k
        self.bottom_k = bottom_k
        self.random_k = random_k
        self.device   = device

        def _ft(val, k):
            return torch.full((k, num_features), val, dtype=torch.float32, device=device)
        def _it(val, k):
            return torch.full((k, num_features), val, dtype=torch.int32,   device=device)

        self.top_activations = [_ft(float('-inf'), top_k)    for _ in range(num_layers)]
        self.top_data_ids    = [_it(-1,             top_k)   for _ in range(num_layers)]
        self.top_token_pos   = [_it(0,              top_k)   for _ in range(num_layers)]

        self.bot_activations = [_ft(float('inf'),  bottom_k) for _ in range(num_layers)]
        self.bot_data_ids    = [_it(-1,             bottom_k) for _ in range(num_layers)]
        self.bot_token_pos   = [_it(0,              bottom_k) for _ in range(num_layers)]

        self.rnd_activations = [torch.zeros(random_k, num_features, dtype=torch.float32, device=device) for _ in range(num_layers)]
        self.rnd_data_ids    = [_it(-1, random_k)  for _ in range(num_layers)]
        self.rnd_token_pos   = [_it(0,  random_k)  for _ in range(num_layers)]

        # Per-feature sample counter for reservoir sampling (CPU int64 — tiny)
        self.rnd_counts = [torch.zeros(num_features, dtype=torch.int64, device=device) for _ in range(num_layers)]

    def update_layer(self, layer: int, sample_acts: torch.Tensor,
                     data_id: int, seq_len: int) -> None:
        """
        Update stored activations for a single (layer, sample) pair.

        Args:
            layer:       Layer index
            sample_acts: float32 tensor of shape (seq_len, num_features) on GPU
            data_id:     Integer identifier for this dataset sample
            seq_len:     Number of valid tokens
        """
        if seq_len == 0:
            return

        k_top = self.top_k
        k_bot = self.bottom_k
        k_rnd = self.random_k
        F     = self.num_features

        # --- Top-K update ---
        # torch.topk over seq_len dim for all features at once: O(seq_len * F) on GPU
        pick_top = min(k_top, seq_len)
        top_vals, top_idx = torch.topk(sample_acts, pick_top, dim=0)  # (pick_top, F)

        combined_acts = torch.cat([self.top_activations[layer], top_vals], dim=0)   # (k_top+pick_top, F)
        combined_ids  = torch.cat([self.top_data_ids[layer],
                                   torch.full((pick_top, F), data_id, dtype=torch.int32, device=self.device)], dim=0)
        combined_pos  = torch.cat([self.top_token_pos[layer],
                                   top_idx.to(torch.int32)], dim=0)

        _, keep = torch.topk(combined_acts, k_top, dim=0)              # (k_top, F)
        self.top_activations[layer] = combined_acts.gather(0, keep)
        self.top_data_ids[layer]    = combined_ids.gather(0, keep)
        self.top_token_pos[layer]   = combined_pos.gather(0, keep)

        # --- Bottom-K update ---
        pick_bot = min(k_bot, seq_len)
        bot_vals, bot_idx = torch.topk(sample_acts, pick_bot, dim=0, largest=False)  # (pick_bot, F)

        combined_acts = torch.cat([self.bot_activations[layer], bot_vals], dim=0)
        combined_ids  = torch.cat([self.bot_data_ids[layer],
                                   torch.full((pick_bot, F), data_id, dtype=torch.int32, device=self.device)], dim=0)
        combined_pos  = torch.cat([self.bot_token_pos[layer],
                                   bot_idx.to(torch.int32)], dim=0)

        _, keep = torch.topk(combined_acts, k_bot, dim=0, largest=False)
        self.bot_activations[layer] = combined_acts.gather(0, keep)
        self.bot_data_ids[layer]    = combined_ids.gather(0, keep)
        self.bot_token_pos[layer]   = combined_pos.gather(0, keep)

        # --- Reservoir random-K update (Algorithm R, vectorized on GPU) ---
        rnd_pos  = torch.randint(0, seq_len, (F,), dtype=torch.int32, device=self.device)  # (F,)
        rnd_acts = sample_acts[rnd_pos, torch.arange(F, device=self.device)]               # (F,)

        counts = self.rnd_counts[layer]   # (F,) int64

        # Phase 1: fill empty slots
        fill_mask     = counts < k_rnd                      # (F,) bool
        fill_features = fill_mask.nonzero(as_tuple=True)[0] # indices
        if fill_features.numel() > 0:
            slots = counts[fill_features]                    # which slot to fill
            self.rnd_activations[layer][slots, fill_features] = rnd_acts[fill_features]
            self.rnd_data_ids[layer][slots, fill_features]    = data_id
            self.rnd_token_pos[layer][slots, fill_features]   = rnd_pos[fill_features]

        # Phase 2: reservoir replacement
        replace_features = (~fill_mask).nonzero(as_tuple=True)[0]
        if replace_features.numel() > 0:
            total = counts[replace_features] + 1             # (R,)
            j     = (torch.rand(replace_features.numel(), device=self.device) * total.float()).long()
            accept_mask = j < k_rnd
            accepted    = replace_features[accept_mask]
            if accepted.numel() > 0:
                slots = j[accept_mask] % k_rnd
                self.rnd_activations[layer][slots, accepted] = rnd_acts[accepted]
                self.rnd_data_ids[layer][slots, accepted]    = data_id
                self.rnd_token_pos[layer][slots, accepted]   = rnd_pos[accepted]

        counts.add_(1)

    def flush_to_db(self, con: duckdb.DuckDBPyConnection) -> None:
        """Move tensors to CPU, build DataFrames, bulk-insert into DuckDB."""
        print("Flushing activation store to database...")
        total_rows = 0

        def _make_df(acts_t, ids_t, pos_t, bucket_name, k, layer):
            # Move to CPU numpy
            acts = acts_t.cpu().numpy()   # (k, F)
            ids  = ids_t.cpu().numpy()
            pos  = pos_t.cpu().numpy()
            F    = acts.shape[1]

            layer_arr = np.full(k * F, layer,  dtype=np.int16)
            feat_arr  = np.repeat(np.arange(F, dtype=np.int32), k)
            acts_flat = acts.T.flatten().astype(np.float32)
            ids_flat  = ids.T.flatten().astype(np.int32)
            pos_flat  = pos.T.flatten().astype(np.int32)

            df = pd.DataFrame({
                'layer':      layer_arr,
                'feature':    feat_arr,
                'data_id':    ids_flat,
                'token_pos':  pos_flat,
                'activation': acts_flat,
                'bucket':     bucket_name,
            })
            df = df[~np.isinf(df['activation'].values)]
            df = df[df['data_id'] >= 0]
            return df

        for layer in tqdm(range(self.num_layers), desc="Flushing layers"):
            frames = [
                _make_df(self.top_activations[layer], self.top_data_ids[layer], self.top_token_pos[layer], 'top',    self.top_k,    layer),
                _make_df(self.bot_activations[layer], self.bot_data_ids[layer], self.bot_token_pos[layer], 'bottom', self.bottom_k, layer),
                _make_df(self.rnd_activations[layer], self.rnd_data_ids[layer], self.rnd_token_pos[layer], 'random', self.random_k, layer),
            ]
            combined = pd.concat(frames, ignore_index=True)
            if len(combined) > 0:
                con.execute("INSERT INTO activations SELECT * FROM combined")
                total_rows += len(combined)

        print(f"Flushed {total_rows:,} rows to activations table.")


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(batch_full_texts, tokenizer, model, device,
                  store, captured_features, num_layers, data_ids, max_length):
    """Tokenize, run model, update ActivationStore for each sample."""
    if not batch_full_texts:
        return

    tokenized = tokenizer(
        batch_full_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids     = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    with torch.no_grad():
        captured_features.clear()
        try:
            _ = model.model(input_ids=input_ids, attention_mask=attention_mask)
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            return

    actual_seq_lens = attention_mask.sum(dim=1).cpu().tolist()

    for layer_num in range(num_layers):
        features = captured_features[layer_num]   # (batch, seq, feat)
        batch_size_actual = features.shape[0]
        for batch_idx in range(batch_size_actual):
            seq_len = int(actual_seq_lens[batch_idx])
            if seq_len == 0:
                continue
            sample_acts = features[batch_idx, :seq_len].float()   # stay on GPU
            store.update_layer(layer_num, sample_acts, data_ids[batch_idx], seq_len)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.db_path)), exist_ok=True)

    # Load model and tokenizer
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    print(f"Loading model on {device} with dtype={dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map=str(device),
        dtype=dtype,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup hooks
    captured_features, mlp_layer_names, cleanup_hooks = setup_hooks(
        model, architecture=args.architecture, capture_stage=args.capture_stage
    )
    num_layers = len(mlp_layer_names)
    print(f"Found {num_layers} MLP layers")

    # Discover num_features via dummy forward pass
    dummy_ids = tokenizer("hello", return_tensors="pt",
                          add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        captured_features.clear()
        _ = model.model(input_ids=dummy_ids)
    num_features = captured_features[0].shape[-1]
    print(f"Detected num_features = {num_features}")

    # Initialize store
    store = ActivationStore(num_layers, num_features,
                            args.top_k, args.bottom_k, args.random_k, device)

    # Initialize database
    con = init_db(args.db_path)
    print(f"Initialized DB at {args.db_path}")

    # Load dataset
    print("Loading dataset...")
    use_streaming = (args.num_samples != -1)
    if use_streaming:
        dataset = load_dataset(args.dataset, split="train", streaming=True)
    else:
        dataset = load_dataset(args.dataset, split="train", streaming=False)

    # Main processing loop
    samples_processed = 0
    global_data_id = 0
    batch_texts = []
    batch_data_ids = []

    total_batches = None
    if args.num_samples != -1:
        total_batches = (args.num_samples + args.batch_size - 1) // args.batch_size

    pbar = tqdm(total=total_batches, desc="Processing batches")

    def flush_batch():
        nonlocal batch_texts, batch_data_ids
        if not batch_texts:
            return
        # Format with chat template
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": t}],
                tokenize=False,
                add_generation_prompt=False,
            )
            for t in batch_texts
        ]
        _insert_dataset_index_batch(con, batch_data_ids, formatted)
        process_batch(formatted, tokenizer, model, device,
                      store, captured_features, num_layers, batch_data_ids, args.max_length)
        batch_texts = []
        batch_data_ids = []
        pbar.update(1)

    for data in dataset:
        if args.num_samples != -1 and samples_processed >= args.num_samples:
            break
        text = data.get('text', '')
        if not text or not text.strip():
            continue
        batch_texts.append(text)
        batch_data_ids.append(global_data_id)
        global_data_id += 1
        samples_processed += 1

        if len(batch_texts) >= args.batch_size:
            flush_batch()

    flush_batch()  # Process remaining samples
    pbar.close()

    # Finalize
    cleanup_hooks()
    store.flush_to_db(con)

    print("Creating index on activations(layer, feature, activation DESC)...")
    con.execute("CREATE INDEX idx_lf ON activations (layer, feature, activation DESC)")

    row_count = con.execute("SELECT COUNT(*) FROM activations").fetchone()[0]
    idx_count = con.execute("SELECT COUNT(*) FROM dataset_index").fetchone()[0]
    print(f"\nDone.")
    print(f"  dataset_index rows : {idx_count:,}")
    print(f"  activations rows   : {row_count:,}")
    print(f"  DB path            : {args.db_path}")

    con.close()


if __name__ == "__main__":
    main()
