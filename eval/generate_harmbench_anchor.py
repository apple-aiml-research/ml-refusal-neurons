#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Standalone HarmBench response generation with anchor-activation hook.

For each prompt:
1. Forward pass (no hook) → capture features[token_pos, feature_idx] = v
2. Generate with hook: features[:, :, feature_idx] = v * m for all tokens

No imports from hooks/ or manipulation/ — fully self-contained.

Usage:
    python eval/generate_harmbench_anchor.py \
        --model Qwen/Qwen3-14B --layer 17 --feature 2154 \
        --token_pos="-5,-6,-7,-8,-9" --token_agg min \
        --m -5.94 --m2 6.74 --best_mult 40 \
        --max_tokens 512 --temperature 0.0
"""

import argparse
import json
import os
import sys
from datetime import datetime
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_jbb_overlap_prompts():
    """
    Return the set of JBB-Behaviors prompt texts, used to exclude HarmBench standard
    prompts that overlap with JBB (200 - 9 = 191 prompts = the paper's validation set).
    """
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    return {row["Goal"].strip() for row in jbb}


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Capture hook: one forward pass to read activation at token_pos
# ---------------------------------------------------------------------------

def capture_activation(model, layer_num, feature_idx, input_ids):
    captured = {}

    def hook_fn(module, inp):
        features = inp[0]
        captured['features'] = features[:, :, feature_idx].detach().clone()
        return None

    for name, module in model.named_modules():
        if name.endswith("mlp") and hasattr(module, 'gate_proj'):
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    if int(parts[i + 1]) == layer_num:
                        handle = module.down_proj.register_forward_pre_hook(hook_fn)
                        with torch.no_grad():
                            model(input_ids)
                        handle.remove()
                        return captured['features']

    raise ValueError(f"Could not find layer {layer_num}")


# ---------------------------------------------------------------------------
# Generation hook: set feature to anchor_value * m for all tokens
# ---------------------------------------------------------------------------

def register_anchor_hook(model, layer_num, feature_idx):
    state = {'active': False, 'anchor_value': 0.0, 'm': 1.0, 'm2': 0.0, 'best_mult': None}

    def pre_down_proj_hook(module, inp):
        if not state['active']:
            return None
        features = inp[0]
        G = state['anchor_value'] * state['m'] + state['m2']
        bm = state['best_mult']
        if bm is not None:
            if bm > 0:
                G = min(G, bm)
            else:
                G = max(G, bm)
        features[:, :, feature_idx] = G
        return (features,)

    for name, module in model.named_modules():
        if name.endswith("mlp") and hasattr(module, 'gate_proj'):
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    if int(parts[i + 1]) == layer_num:
                        handle = module.down_proj.register_forward_pre_hook(pre_down_proj_hook)
                        return handle, state, name

    raise ValueError(f"Could not find layer {layer_num}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _build_input_text(tokenizer, prompt, enable_thinking=False):
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def generate(model, tokenizer, prompt, max_tokens, temperature,
             hook_state, active, enable_thinking=False,
             layer=None, feature=None, token_pos=None, m=None, m2=None,
             best_mult=None, token_agg=None):
    """
    If active=True: capture anchor value first, then generate with hook.
    If active=False: generate baseline (no hook).
    token_pos: single int or list of ints.
    token_agg: 'min' or 'max' when multiple token positions.
    """
    text = _build_input_text(tokenizer, prompt, enable_thinking=enable_thinking)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    anchor_val = None
    if active and layer is not None:
        hook_state['active'] = False
        all_activations = capture_activation(model, layer, feature, input_ids)
        seq_len = input_ids.shape[1]

        if isinstance(token_pos, list):
            vals = []
            for tp in token_pos:
                ap = tp if tp >= 0 else seq_len + tp
                if 0 <= ap < seq_len:
                    vals.append(all_activations[0, ap].item())
            if vals:
                anchor_val = min(vals) if token_agg == 'min' else max(vals)
            else:
                anchor_val = 0.0
        else:
            abs_pos = token_pos if token_pos >= 0 else seq_len + token_pos
            if 0 <= abs_pos < seq_len:
                anchor_val = all_activations[0, abs_pos].item()
            else:
                anchor_val = 0.0

        hook_state['anchor_value'] = anchor_val
        hook_state['m'] = m
        hook_state['m2'] = m2 if m2 is not None else 0.0
        hook_state['best_mult'] = best_mult

    hook_state['active'] = active

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    hook_state['active'] = False

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    if full.startswith(prompt_text):
        response = full[len(prompt_text):].strip()
    else:
        response = full

    return response, anchor_val


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_harmbench_dataset(config="standard"):
    print(f"Loading HarmBench dataset (config: {config})...")
    dataset = load_dataset("walledai/HarmBench", config, split="train")
    print(f"Loaded {len(dataset)} examples")
    return dataset


def collect_examples(dataset, limit=None, indices=None):
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    examples = []
    for idx, example in enumerate(dataset):
        prompt = None
        for field in ['Behavior', 'prompt', 'text', 'input', 'behavior']:
            if field in example:
                prompt = example[field]
                break
        if prompt is None:
            continue
        examples.append((idx, prompt, example))
    if indices:
        keep = set(int(x.strip()) for x in indices.split(","))
        examples = [(idx, p, ex) for idx, p, ex in examples if idx in keep]
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Standalone HarmBench response generation with anchor-activation hook")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--feature", type=int, required=True)
    parser.add_argument("--token_pos", type=str, required=True,
                        help="Token position(s) to anchor (negative = from end). "
                             "Single: -3. Multiple: -3,-5,-8")
    parser.add_argument("--token_agg", type=str, default=None, choices=["min", "max"],
                        help="Aggregation for multiple token positions (required if multiple)")
    parser.add_argument("--m", type=float, required=True, help="Multiplier on captured activation")
    parser.add_argument("--m2", type=float, default=0.0, help="Additive constant")
    parser.add_argument("--best_mult", type=float, default=None,
                        help="Clamp G: min(G, best_mult) if best_mult>0, max(G, best_mult) if best_mult<0")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--config", type=str, default="standard",
                        choices=["standard", "contextual", "copyright"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--full_harmbench", action="store_true",
                        help="Run on the full 200-prompt HarmBench standard set instead of "
                             "the paper's HarmBench-191 validation set (standard minus JBB-Behaviors overlap).")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--baseline_cache", type=str, default=None)
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--think", action="store_true")
    args = parser.parse_args()

    # Parse token_pos: single int or comma-separated list
    tp_parts = [int(x.strip()) for x in args.token_pos.split(",")]
    token_pos = tp_parts if len(tp_parts) > 1 else tp_parts[0]
    token_agg = args.token_agg

    # Output path
    if args.output is None:
        model_name = (args.model_name or args.model).split("/")[-1]
        out_dir = args.output_dir or os.path.join(REPO_ROOT, "results", "eval", "harmbench")
        os.makedirs(out_dir, exist_ok=True)
        parts = [
            f"harmbench_anchor_{model_name}",
            f"L{args.layer}",
            f"F{args.feature}",
            f"TP{args.token_pos.replace(',','_')}",
            f"M{args.m}",
            f"M2_{args.m2}",
            f"BM_{args.best_mult}" if args.best_mult is not None else None,
        ]
        parts = [p for p in parts if p is not None]
        if args.think:
            parts.append("TK")
        args.output = os.path.join(out_dir, f"{'_'.join(parts)}.json")

    print("=" * 80)
    print("HARMBENCH EVALUATION — ANCHOR ACTIVATION")
    print("=" * 80)
    print(f"Model:      {args.model}")
    print(f"Layer:      {args.layer}   Feature: {args.feature}")
    print(f"token_pos:  {token_pos}")
    if token_agg:
        print(f"token_agg:  {token_agg}")
    print(f"m:          {args.m}")
    print(f"m2:         {args.m2}")
    print(f"best_mult:  {args.best_mult}")
    print(f"Max tokens: {args.max_tokens}   Temperature: {args.temperature}")
    print(f"Output:     {args.output}")
    print("=" * 80)
    print()

    # Load dataset
    dataset = load_harmbench_dataset(config=args.config)
    examples = collect_examples(dataset, limit=args.limit, indices=args.indices)
    if not args.full_harmbench and args.config == "standard":
        jbb_prompts = load_jbb_overlap_prompts()
        before = len(examples)
        examples = [(idx, p, ex) for idx, p, ex in examples if p.strip() not in jbb_prompts]
        print(f"HarmBench-191 filter (standard minus JBB-Behaviors overlap): "
              f"kept {len(examples)}/{before} prompts. Pass --full_harmbench to run on all {before}.")
        if args.limit is None and not args.indices:
            assert len(examples) == 191, (
                f"Expected 191 HarmBench-191 prompts after JBB-overlap filtering, "
                f"got {len(examples)}. HarmBench standard or JBB-Behaviors may have "
                f"changed upstream — pass --full_harmbench to bypass this check.")
    print(f"Processing {len(examples)} examples\n")

    # Load baseline cache
    baseline_cache_all = {}
    cache_key = f"T{args.temperature}_MT{args.max_tokens}"
    if args.baseline_cache and os.path.exists(args.baseline_cache):
        with open(args.baseline_cache) as f:
            baseline_cache_all = json.load(f)
        n = len(baseline_cache_all.get(cache_key, {}))
        print(f"Loaded baseline cache: {n} entries for {cache_key}")
    prompt_cache = baseline_cache_all.setdefault(cache_key, {})

    # Load model
    print(f"Loading model: {args.model} ...")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
            else (torch.float16 if torch.cuda.is_available() else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    ).eval()
    model.generation_config.max_length = None
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    hook_handle, hook_state, layer_name = register_anchor_hook(
        model, args.layer, args.feature)
    print(f"Hook: layer={args.layer} feature={args.feature} token_pos={token_pos} "
          f"m={args.m} m2={args.m2} token_agg={token_agg} ({layer_name})")

    # Show what token(s) token_pos maps to using the first prompt
    first_prompt = examples[0][1]
    first_text = _build_input_text(tokenizer, first_prompt, enable_thinking=args.think)
    first_ids = tokenizer(first_text, return_tensors="pt").input_ids
    seq_len = first_ids.shape[1]
    positions = token_pos if isinstance(token_pos, list) else [token_pos]
    for tp in positions:
        ap = tp if tp >= 0 else seq_len + tp
        if 0 <= ap < seq_len:
            tid = first_ids[0, ap].item()
            tok = tokenizer.decode([tid])
            print(f"token_pos={tp} -> token[{ap}/{seq_len}] id={tid} repr={repr(tok)}")
        else:
            print(f"WARNING: token_pos={tp} out of range (seq_len={seq_len})")
    print()

    # Generate
    results = []
    new_cache = {}
    anchor_values = []

    for idx, prompt, example in tqdm(examples, desc="Generating"):
        try:
            baseline_response = None
            if not args.skip_baseline:
                if prompt in prompt_cache:
                    baseline_response = prompt_cache[prompt]
                else:
                    baseline_response, _ = generate(
                        model, tokenizer, prompt, args.max_tokens, args.temperature,
                        hook_state, active=False, enable_thinking=args.think)
                    new_cache[prompt] = baseline_response

            anchored_response, anchor_val = generate(
                model, tokenizer, prompt, args.max_tokens, args.temperature,
                hook_state, active=True, enable_thinking=args.think,
                layer=args.layer, feature=args.feature,
                token_pos=token_pos, m=args.m, m2=args.m2,
                best_mult=args.best_mult, token_agg=token_agg)

            if anchor_val is not None:
                anchor_values.append(anchor_val)

            result = {
                "idx": idx,
                "prompt": prompt,
                "baseline_response": baseline_response,
                "ablated_response": anchored_response,
                "anchor_value": anchor_val,
                "effective_value": anchor_val * args.m + args.m2 if anchor_val is not None else None,
            }
            for k, v in example.items():
                if k not in result:
                    result[k] = v
            results.append(result)

        except Exception as e:
            print(f"\nError on example {idx}: {e}")
            result = {
                "idx": idx,
                "prompt": prompt,
                "baseline_response": None,
                "ablated_response": None,
                "anchor_value": None,
                "error": str(e),
            }
            for k, v in example.items():
                if k not in result:
                    result[k] = v
            results.append(result)

    hook_handle.remove()

    # Update baseline cache
    prompt_cache.update(new_cache)
    if args.baseline_cache and new_cache:
        os.makedirs(os.path.dirname(args.baseline_cache) or ".", exist_ok=True)
        with open(args.baseline_cache, "w") as f:
            json.dump(baseline_cache_all, f, indent=2)
        print(f"Updated baseline cache with {len(new_cache)} new entries")

    # Anchor stats
    if anchor_values:
        import statistics
        print(f"\nAnchor value stats (n={len(anchor_values)}):")
        print(f"  mean={statistics.mean(anchor_values):.4f}  "
              f"median={statistics.median(anchor_values):.4f}  "
              f"min={min(anchor_values):.4f}  max={max(anchor_values):.4f}  "
              f"stdev={statistics.stdev(anchor_values):.4f}" if len(anchor_values) > 1
              else f"  value={anchor_values[0]:.4f}")

    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
                exist_ok=True)

    output_data = {
        "metadata": {
            "model": args.model_name or args.model,
            "layer": args.layer,
            "feature": args.feature,
            "token_pos": args.token_pos,
            "token_agg": token_agg,
            "m": args.m,
            "m2": args.m2,
            "best_mult": args.best_mult,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "think": args.think,
            "config": args.config,
            "harmbench_191_filtered": (not args.full_harmbench and args.config == "standard"),
            "hook_formula": "features[:, :, f] = anchor_value * m"
                            " where anchor_value = features[0, token_pos, f] from clean forward pass",
            "timestamp": datetime.now().isoformat(),
            "total_examples": len(results),
            "successful": sum(1 for r in results if r.get("ablated_response") is not None),
            "failed": sum(1 for r in results if r.get("ablated_response") is None),
        },
        "results": results,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(results)} results to {args.output}")
    print(f"  Successful: {output_data['metadata']['successful']}")
    print(f"  Failed:     {output_data['metadata']['failed']}")
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
