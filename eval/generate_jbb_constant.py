#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Generate constant neuron intervention responses on JailbreakBench/JBB-Behaviors.

For each prompt: generate a baseline response (no hook), then an intervened
response (hook active), and save both to JSON.

Usage:
    python eval/generate_jbb_constant.py \
        --model Qwen/Qwen3-32B \
        --layer 40 --feature 15515 --multiplier -80
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
from hooks.model_hooks import detect_architecture
from hooks.amplify_hooks import register_amplify_hook


def load_jbb_dataset(split="harmful"):
    print(f"Loading JBB-Behaviors dataset (split: {split})...")
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split=split)
    ds = ds.map(lambda x, idx: {**x, "original_idx": idx}, with_indices=True)
    print(f"  Loaded {len(ds)} examples")
    return ds


def generate(model, tokenizer, prompt, max_tokens, temperature, device, enable_thinking=False):
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate constant neuron intervention responses on JBB-Behaviors"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--feature", type=int, required=True)
    parser.add_argument("--multiplier", type=float, required=True)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--architecture", type=str, default="auto",
                        choices=["auto", "llama", "gpt2"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="harmful",
                        choices=["harmful", "benign"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_pre_down_proj", dest="pre_down_proj", action="store_false", default=True,
                        help="Use the post-hook mode instead of hooking down_proj input directly "
                             "(default: pre_down_proj on — the mode used throughout the paper).")
    parser.add_argument("--additive", dest="additive", action="store_true", default=False,
                        help="Add multiplier to the activation instead of replacing it "
                             "(used for concept-neuron amplification; default: off, matches "
                             "refusal-neuron generation in the paper).")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--think", action="store_true")
    args = parser.parse_args()

    additive = args.additive

    if args.output is None:
        model_name = args.model.split("/")[-1]
        filename = f"jbb_{model_name}_L{args.layer}_F{args.feature}_M{args.multiplier}.json"
        out_dir = args.output_dir or os.path.join(REPO_ROOT, "results", "eval", "jbb")
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, filename)

    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Layer:      {args.layer}   Feature: {args.feature}   Mult: {args.multiplier}")
    print(f"Output:     {args.output}")
    print("=" * 60)

    dataset = load_jbb_dataset(split=args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.architecture == "auto":
        args.architecture = detect_architecture(model)

    hook_handle, hook_state, _ = register_amplify_hook(
        model, args.architecture, args.layer, args.feature, args.multiplier,
        pre_down_proj=args.pre_down_proj, additive=additive,
        response_mult=args.multiplier)

    results = []
    for idx, example in enumerate(tqdm(dataset, desc="Generating")):
        prompt = None
        for field in ["Goal", "goal", "Behavior", "prompt", "text", "input", "behavior"]:
            if field in example:
                prompt = example[field]
                break
        if prompt is None:
            print(f"Warning: no prompt field in example {idx}")
            continue

        try:
            baseline_response = None
            if not args.skip_baseline:
                hook_state["active"] = False
                baseline_response = generate(
                    model, tokenizer, prompt, args.max_tokens,
                    args.temperature, device, args.think)

            hook_state["active"] = True
            ablated_response = generate(
                model, tokenizer, prompt, args.max_tokens,
                args.temperature, device, args.think)

            result = {"idx": idx, "prompt": prompt,
                      "baseline_response": baseline_response,
                      "ablated_response": ablated_response}
            for k, v in example.items():
                if k not in result:
                    result[k] = v
            results.append(result)

        except Exception as e:
            print(f"\nError at example {idx}: {e}")
            result = {"idx": idx, "prompt": prompt,
                      "baseline_response": None, "ablated_response": None, "error": str(e)}
            for k, v in example.items():
                if k not in result:
                    result[k] = v
            results.append(result)

    hook_handle.remove()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    output_data = {
        "metadata": {
            "model": args.model, "layer": args.layer, "feature": args.feature,
            "multiplier": args.multiplier, "max_tokens": args.max_tokens,
            "temperature": args.temperature, "architecture": args.architecture,
            "pre_down_proj": args.pre_down_proj, "additive": additive,
            "dataset": "JailbreakBench/JBB-Behaviors", "split": args.split,
            "timestamp": datetime.now().isoformat(),
            "total_examples": len(results),
            "successful": sum(1 for r in results if r.get("ablated_response") is not None),
            "failed": sum(1 for r in results if r.get("ablated_response") is None),
        },
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(results)} results to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
