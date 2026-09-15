# Max Activations

Tools for finding which tokens (with surrounding context) produce a given MLP neuron's highest and lowest activation values, across a large text corpus. Useful for interpreting candidate concept/refusal neurons.

The pipeline has two pieces:

1. **`build_activation_db.py`** — runs the model on a large text dataset, records the top-K / bottom-K / random-K activations for every neuron in every MLP layer, and stores them in a DuckDB file.
2. **`feature_server.py` + `feature_viewer.html`** — small HTTP server + browser UI that queries the DB and renders decoded token context so you can visually inspect what a neuron responds to.

---

## Step 1 — Build the activation database

Runs the model on `--num_samples` prompts from a HuggingFace dataset (default: `monology/pile-uncopyrighted`), captures MLP activations, and writes the top / bottom / random activations per neuron to a `.duckdb` file.

```bash
python max_activations/build_activation_db.py \
    --model Qwen/Qwen3-1.7B \
    --num_samples 20000 \
    --batch_size 8
```

By default the DB is written to `max_activations/<model_slug>.duckdb` (e.g. `Qwen_Qwen3-1.7B.duckdb`). Use `--db_path` to override. The paper uses `--num_samples 20000`.

Key flags:

| Flag | Description |
|---|---|
| `--model` | HuggingFace model ID |
| `--num_samples` | Number of dataset samples to run through the model (`-1` = all) |
| `--top_k` / `--bottom_k` | How many highest / lowest activations to keep per neuron (default: 200 each) |
| `--random_k` | Reservoir-sampled random activations per neuron (default: 50) |
| `--dataset` | HuggingFace dataset to run the model on (default: `monology/pile-uncopyrighted`) |
| `--max_length` | Max tokens per sample (default: 512) |
| `--capture_stage` | Which MLP stage to record: `default` (= `act_fn(gate_proj(x)) * up_proj(x)`, the actual neuron value `h_i` used everywhere else in this repo), `up_proj`, `gate_proj`, or `down_proj`. |
| `--batch_size` | Model inference batch size |

---

## Step 2 — Browse the database with the UI

Start the HTTP server:

```bash
python max_activations/feature_server.py \
    --db max_activations/Qwen_Qwen3-1.7B.duckdb \
    --model Qwen/Qwen3-1.7B \
    --port 8001 --context_window 20
```

Then open `max_activations/feature_viewer.html` in your browser. The page talks to `localhost:8001` by default.

The UI lets you pick a `(layer, feature)` pair and see:

- **Top activations** — the token positions (with surrounding context) where this neuron fires most strongly.
- **Bottom activations** — the token positions where it is most negative.
- **Random activations** — reservoir-sampled activations, useful for spotting whether the neuron is typically silent or typically active.

Server flags:

| Flag | Description |
|---|---|
| `--db` | Path to the `.duckdb` file |
| `--model` | HuggingFace model ID for the tokenizer (default: inferred from DB filename) |
| `--port` | HTTP port (default: 8001) |
| `--top_n` | Rows returned per bucket (default: 20) |
| `--context_window` | Tokens of context shown on each side (default: 20) |
| `--max_length` | Re-tokenization limit for decoding context — must match the `--max_length` used when building the DB (default: 512) |

---

## Step 3 (optional) — Query from Python

`query_db.py` exposes the same query helpers used by the server, callable from any Python script:

```python
from transformers import AutoTokenizer
from max_activations.query_db import get_top_activations

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
rows = get_top_activations(
    db_path="max_activations/Qwen_Qwen3-1.7B.duckdb",
    layer=20, feature=4256,
    n=10,
    tokenizer=tokenizer,
    context_window=20,
)
for r in rows:
    print(f"{r['activation']:+.3f}  ...{r['context_before']}[{r['token']}]{r['context_after']}...")
```

`query_db.py` also exposes `get_bottom_activations`, `get_spectrum` (top + bottom + random combined), and `get_feature_stats` (min/max/mean per feature). `get_top_activations`/`get_bottom_activations`'s `n` should be ≤ the `--top_k`/`--bottom_k` the DB was built with.
