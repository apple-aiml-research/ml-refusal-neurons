#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Standalone JBB-Behaviors response generation with anchor-activation hook.

For each prompt:
1. Forward pass (no hook) -> capture features[token_pos, feature_idx] = v
2. Generate with hook: features[:, :, feature_idx] = clamp(v * m + m2, best_mult)

No imports from hooks/ or manipulation/ — fully self-contained.

Usage:
    python eval/generate_jbb_anchor.py \
        --model Qwen/Qwen3-14B --layer 17 --feature 2154 \
        --token_pos="-5,-6,-7,-8,-9" --token_agg min \
        --m -5.936656 --m2 6.7378 --best_mult 40 \
        --max_tokens 512 --temperature 0.0 --skip_baseline \
        --output_dir results/eval/jbb/qwen14b
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


def _build_input_text(tokenizer, prompt, enable_thinking=False):
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, prompt, max_tokens, temperature,
             hook_state, active, enable_thinking=False,
             layer=None, feature=None, token_pos=None, m=None, m2=None,
             best_mult=None, token_agg=None):
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
            anchor_val = (min(vals) if token_agg == 'min' else max(vals)) if vals else 0.0
        else:
            abs_pos = token_pos if token_pos >= 0 else seq_len + token_pos
            anchor_val = all_activations[0, abs_pos].item() if 0 <= abs_pos < seq_len else 0.0

        hook_state['anchor_value'] = anchor_val
        hook_state['m'] = m
        hook_state['m2'] = m2 if m2 is not None else 0.0
        hook_state['best_mult'] = best_mult

    hook_state['active'] = active

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id)

    hook_state['active'] = False

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    response = full[len(prompt_text):].strip() if full.startswith(prompt_text) else full
    return response, anchor_val


def load_jbb_dataset(split="harmful"):
    print(f"Loading JBB-Behaviors dataset (split: {split})...")
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split=split)
    print(f"Loaded {len(ds)} examples")
    return ds


def collect_examples(dataset, limit=None):
    examples = []
    for idx, ex in enumerate(dataset):
        prompt = None
        for field in ['Goal', 'goal', 'Behavior', 'prompt', 'text', 'input', 'behavior']:
            if field in ex:
                prompt = ex[field]
                break
        if prompt:
            examples.append((idx, prompt, ex))
    if limit:
        examples = examples[:limit]
    return examples


def main():
    parser = argparse.ArgumentParser(description="JBB-Behaviors response generation with anchor hook")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--feature", type=int, required=True)
    parser.add_argument("--token_pos", type=str, required=True,
                        help="Token position(s). Single: -6. Multiple: -5,-6,-7,-8,-9")
    parser.add_argument("--token_agg", type=str, default=None, choices=["min", "max"])
    parser.add_argument("--m", type=float, required=True)
    parser.add_argument("--m2", type=float, default=0.0)
    parser.add_argument("--best_mult", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--split", type=str, default="harmful", choices=["harmful", "benign"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--baseline_cache", type=str, default=None)
    parser.add_argument("--think", action="store_true")
    args = parser.parse_args()

    tp_parts = [int(x.strip()) for x in args.token_pos.split(",")]
    token_pos = tp_parts if len(tp_parts) > 1 else tp_parts[0]
    token_agg = args.token_agg

    if args.output is None:
        model_name = (args.model_name or args.model).split("/")[-1]
        out_dir = args.output_dir or os.path.join(REPO_ROOT, "results", "eval", "jbb")
        os.makedirs(out_dir, exist_ok=True)
        parts = [p for p in [
            f"jbb_anchor_{model_name}",
            f"L{args.layer}", f"F{args.feature}",
            f"TP{args.token_pos.replace(',','_')}",
            f"M{args.m}", f"M2_{args.m2}",
            f"BM_{args.best_mult}" if args.best_mult is not None else None,
        ] if p is not None]
        args.output = os.path.join(out_dir, f"{'_'.join(parts)}.json")

    print("=" * 80)
    print("JBB-BEHAVIORS EVALUATION — ANCHOR ACTIVATION")
    print("=" * 80)
    print(f"Model:      {args.model}")
    print(f"Layer:      {args.layer}   Feature: {args.feature}")
    print(f"token_pos:  {token_pos}   token_agg: {token_agg}")
    print(f"m:          {args.m}   m2: {args.m2}   best_mult: {args.best_mult}")
    print(f"Split:      {args.split}")
    print(f"Output:     {args.output}")
    print("=" * 80)

    dataset = load_jbb_dataset(split=args.split)
    examples = collect_examples(dataset, limit=args.limit)
    print(f"Processing {len(examples)} examples\n")

    # Load baseline cache
    baseline_cache_all = {}
    cache_key = f"T{args.temperature}_MT{args.max_tokens}"
    if args.baseline_cache and os.path.exists(args.baseline_cache):
        with open(args.baseline_cache) as f:
            baseline_cache_all = json.load(f)
    prompt_cache = baseline_cache_all.setdefault(cache_key, {})

    # Load model
    print(f"Loading model: {args.model} ...")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True).eval()
    model.generation_config.max_length = None
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    hook_handle, hook_state, layer_name = register_anchor_hook(model, args.layer, args.feature)
    print(f"Hook: {layer_name}")

    # Show tokens
    first_text = _build_input_text(tokenizer, examples[0][1], enable_thinking=args.think)
    first_ids = tokenizer(first_text, return_tensors="pt").input_ids
    seq_len = first_ids.shape[1]
    positions = token_pos if isinstance(token_pos, list) else [token_pos]
    if token_agg:
        print(f"Aggregation: {token_agg} across {positions}")
    for tp in positions:
        ap = tp if tp >= 0 else seq_len + tp
        if 0 <= ap < seq_len:
            tid = first_ids[0, ap].item()
            tok = tokenizer.decode([tid])
            print(f"  token_pos={tp} -> token[{ap}/{seq_len}] id={tid} repr={repr(tok)}")
    print()

    # Generate
    results = []
    new_cache = {}

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

            result = {
                "idx": idx, "prompt": prompt,
                "baseline_response": baseline_response,
                "ablated_response": anchored_response,
                "anchor_value": anchor_val,
            }
            for k, v in example.items():
                if k not in result:
                    result[k] = v
            results.append(result)

        except Exception as e:
            print(f"\nError on example {idx}: {e}")
            results.append({
                "idx": idx, "prompt": prompt,
                "baseline_response": None, "ablated_response": None,
                "anchor_value": None, "error": str(e),
            })

    hook_handle.remove()

    # Update cache
    prompt_cache.update(new_cache)
    if args.baseline_cache and new_cache:
        os.makedirs(os.path.dirname(args.baseline_cache) or ".", exist_ok=True)
        with open(args.baseline_cache, "w") as f:
            json.dump(baseline_cache_all, f, indent=2)

    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    output_data = {
        "metadata": {
            "model": args.model_name or args.model,
            "layer": args.layer, "feature": args.feature,
            "token_pos": args.token_pos, "token_agg": token_agg,
            "m": args.m, "m2": args.m2, "best_mult": args.best_mult,
            "max_tokens": args.max_tokens, "temperature": args.temperature,
            "split": args.split,
            "hook_formula": "G = anchor_value * m + m2, clamped by best_mult",
            "timestamp": datetime.now().isoformat(),
            "total_examples": len(results),
            "successful": sum(1 for r in results if r.get("ablated_response") is not None),
        },
        "results": results,
    }
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(results)} results to {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
