#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Generate constant neuron intervention responses on HarmBench.

For each prompt: generate a baseline response (no hook), then an intervened
response (hook active), and save both to JSON.

Usage:
    python eval/generate_harmbench_constant.py \
        --model Qwen/Qwen3-32B \
        --layer 40 --feature 15515 --multiplier -80
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
from hooks.model_hooks import detect_architecture
from hooks.amplify_hooks import register_amplify_hook, parse_mult_intervals


def load_jbb_overlap_prompts():
    """
    Return the set of JBB-Behaviors prompt texts, used to exclude HarmBench standard
    prompts that overlap with JBB (200 - 9 = 191 prompts = the paper's validation set).
    """
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    return {row["Goal"].strip() for row in jbb}


def load_harmbench_dataset(config="standard"):
    print(f"Loading HarmBench dataset (config: {config})...")
    dataset = load_dataset("walledai/HarmBench", config, split="train")
    print(f"Loaded {len(dataset)} examples")
    return dataset


def _collect_examples(dataset, limit=None):
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    examples = []
    for idx, example in enumerate(dataset):
        prompt = None
        for field in ["Behavior", "prompt", "text", "input", "behavior"]:
            if field in example:
                prompt = example[field]
                break
        if prompt is None:
            print(f"Warning: no prompt field in example {idx}")
            continue
        examples.append((idx, prompt, example))
    return examples


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


def _run_generation_loop(model, tokenizer, hook_state, examples, args, prompt_cache, device):
    results = []
    new_cache = {}
    for idx, prompt, example in tqdm(examples, desc="Generating"):
        try:
            if args.skip_baseline:
                baseline_response = None
            elif prompt in prompt_cache:
                baseline_response = prompt_cache[prompt]
            else:
                hook_state["active"] = False
                baseline_response = generate(
                    model, tokenizer, prompt, args.max_tokens,
                    args.temperature, device, args.think)
                new_cache[prompt] = baseline_response

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

    return results, new_cache


def _run_subprocess_worker(args, p_intervals, r_intervals):
    dataset = load_harmbench_dataset(config=args.config)
    all_examples = _collect_examples(dataset, limit=args.limit)
    if args.indices:
        keep = set(int(x.strip()) for x in args.indices.split(","))
        all_examples = [(idx, p, ex) for idx, p, ex in all_examples if idx in keep]
    if not args.full_harmbench and args.config == "standard":
        jbb_prompts = load_jbb_overlap_prompts()
        all_examples = [(idx, p, ex) for idx, p, ex in all_examples if p.strip() not in jbb_prompts]
        if args.limit is None and not args.indices:
            assert len(all_examples) == 191, (
                f"Expected 191 HarmBench-191 prompts after JBB-overlap filtering, "
                f"got {len(all_examples)}. HarmBench standard or JBB-Behaviors may have "
                f"changed upstream — pass --full_harmbench to bypass this check.")

    offset = args._subprocess_offset
    count = args._subprocess_count if args._subprocess_count is not None else len(all_examples)
    my_examples = all_examples[offset:offset + count]

    cached_prompts = {}
    if args._subprocess_cache_input and os.path.exists(args._subprocess_cache_input):
        with open(args._subprocess_cache_input) as f:
            cached_prompts = json.load(f)

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch_dtype, device_map="auto",
        trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    architecture = args.architecture
    if architecture == "auto":
        architecture = detect_architecture(model)

    hook_handle, hook_state, _ = register_amplify_hook(
        model, architecture, args.layer, args.feature, args.multiplier,
        pre_down_proj=args.pre_down_proj, additive=args.additive,
        response_mult=args.multiplier,
        p_intervals=p_intervals, r_intervals=r_intervals)

    device = torch.device("cuda:0")
    results, new_cache = _run_generation_loop(
        model, tokenizer, hook_state, my_examples, args, cached_prompts, device)
    hook_handle.remove()

    with open(args._subprocess_output, "w") as f:
        json.dump({"results": results, "new_cache": new_cache}, f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate constant neuron intervention responses on HarmBench"
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
    parser.add_argument("--config", type=str, default="standard",
                        choices=["standard", "contextual", "copyright"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_pre_down_proj", dest="pre_down_proj", action="store_false", default=True,
                        help="Use the post-hook mode instead of hooking down_proj input directly "
                             "(default: pre_down_proj on — the mode used throughout the paper).")
    parser.add_argument("--additive", dest="additive", action="store_true", default=False,
                        help="Add multiplier to the activation instead of replacing it "
                             "(used for concept-neuron amplification; default: off, matches "
                             "refusal-neuron generation in the paper).")
    parser.add_argument("--p_mult", type=str, default=None)
    parser.add_argument("--r_mult", type=str, default=None)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--baseline_cache", type=str, default=None)
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline (no-hook) generation entirely -- baseline_response "
                             "is left null for every example. Use when the baseline is not "
                             "needed (e.g. only comparing ablated ASR across candidates) or is "
                             "already available from a prior run.")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--full_harmbench", action="store_true",
                        help="Run on the full 200-prompt HarmBench standard set instead of "
                             "the paper's HarmBench-191 validation set (standard minus JBB-Behaviors overlap).")
    parser.add_argument("--num_gpus", type=int, default=1)
    # Hidden subprocess args
    parser.add_argument("--_subprocess_output", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_subprocess_cache_input", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_subprocess_offset", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_subprocess_count", type=int, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    p_intervals = parse_mult_intervals(args.p_mult)
    r_intervals = parse_mult_intervals(args.r_mult)

    if args._subprocess_output is not None:
        _run_subprocess_worker(args, p_intervals, r_intervals)
        return 0

    if args.output is None:
        model_name = (args.model_name or args.model).split("/")[-1]
        out_dir = args.output_dir or os.path.join(REPO_ROOT, "results", "eval", "harmbench")
        os.makedirs(out_dir, exist_ok=True)
        parts = [f"harmbench_{model_name}", f"L{args.layer}", f"F{args.feature}",
                 f"M{args.multiplier}"]
        args.output = os.path.join(out_dir, "_".join(parts) + ".json")

    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Layer:      {args.layer}   Feature: {args.feature}   Mult: {args.multiplier}")
    print(f"Output:     {args.output}")
    print("=" * 60)

    dataset = load_harmbench_dataset(config=args.config)
    all_examples = _collect_examples(dataset, limit=args.limit)
    if args.indices:
        keep = set(int(x.strip()) for x in args.indices.split(","))
        all_examples = [(idx, p, ex) for idx, p, ex in all_examples if idx in keep]
    if not args.full_harmbench and args.config == "standard":
        jbb_prompts = load_jbb_overlap_prompts()
        before = len(all_examples)
        all_examples = [(idx, p, ex) for idx, p, ex in all_examples if p.strip() not in jbb_prompts]
        print(f"HarmBench-191 filter (standard minus JBB-Behaviors overlap): "
              f"kept {len(all_examples)}/{before} prompts. Pass --full_harmbench to run on all {before}.")
        if args.limit is None and not args.indices:
            assert len(all_examples) == 191, (
                f"Expected 191 HarmBench-191 prompts after JBB-overlap filtering, "
                f"got {len(all_examples)}. HarmBench standard or JBB-Behaviors may have "
                f"changed upstream — pass --full_harmbench to bypass this check.")

    baseline_cache = {}
    cache_key = f"T{args.temperature}_MT{args.max_tokens}"
    if args.baseline_cache and os.path.exists(args.baseline_cache):
        with open(args.baseline_cache) as f:
            baseline_cache = json.load(f)
    prompt_cache = baseline_cache.setdefault(cache_key, {})

    additive = args.additive

    if args.num_gpus > 1:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        all_gpu_ids = ([int(x.strip()) for x in cuda_visible.split(",")]
                       if cuda_visible else list(range(torch.cuda.device_count())))
        n_gpus = min(args.num_gpus, len(all_gpu_ids))
        gpu_ids = all_gpu_ids[:n_gpus]

        tmpdir = tempfile.mkdtemp(prefix="harmbench_mp_")
        cache_file = os.path.join(tmpdir, "prompt_cache.json")
        with open(cache_file, "w") as f:
            json.dump(prompt_cache, f)

        chunk_size = (len(all_examples) + n_gpus - 1) // n_gpus
        procs, output_files = [], []
        for rank in range(n_gpus):
            offset = rank * chunk_size
            count = min(chunk_size, len(all_examples) - offset)
            if count <= 0:
                break
            out_file = os.path.join(tmpdir, f"rank_{rank}.json")
            output_files.append(out_file)
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--model", args.model,
                   "--layer", str(args.layer), "--feature", str(args.feature),
                   "--multiplier", str(args.multiplier),
                   "--max_tokens", str(args.max_tokens),
                   "--temperature", str(args.temperature),
                   "--architecture", args.architecture,
                   "--config", args.config,
                   "--_subprocess_output", out_file,
                   "--_subprocess_cache_input", cache_file,
                   "--_subprocess_offset", str(offset),
                   "--_subprocess_count", str(count)]
            if args.limit is not None:
                cmd += ["--limit", str(args.limit)]
            if not args.pre_down_proj:
                cmd += ["--no_pre_down_proj"]
            if args.additive:
                cmd += ["--additive"]
            if args.p_mult:
                cmd += ["--p_mult", args.p_mult]
            if args.r_mult:
                cmd += ["--r_mult", args.r_mult]
            if args.think:
                cmd += ["--think"]
            if args.indices:
                cmd += ["--indices", args.indices]
            if args.full_harmbench:
                cmd += ["--full_harmbench"]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[rank])
            procs.append(subprocess.Popen(cmd, env=env))

        for proc in procs:
            proc.wait()

        all_results, all_new_cache = [], {}
        for out_file in output_files:
            if os.path.exists(out_file):
                with open(out_file) as f:
                    data = json.load(f)
                all_results.extend(data["results"])
                all_new_cache.update(data["new_cache"])
        shutil.rmtree(tmpdir)

        prompt_cache.update(all_new_cache)
        if args.baseline_cache:
            os.makedirs(os.path.dirname(args.baseline_cache) or ".", exist_ok=True)
            with open(args.baseline_cache, "w") as f:
                json.dump(baseline_cache, f, indent=2)

        results = sorted(all_results, key=lambda r: r["idx"])

    else:
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
            response_mult=args.multiplier,
            p_intervals=p_intervals, r_intervals=r_intervals)

        results, new_cache = _run_generation_loop(
            model, tokenizer, hook_state, all_examples, args, prompt_cache, device)

        prompt_cache.update(new_cache)
        if args.baseline_cache and new_cache:
            os.makedirs(os.path.dirname(args.baseline_cache) or ".", exist_ok=True)
            with open(args.baseline_cache, "w") as f:
                json.dump(baseline_cache, f, indent=2)

        hook_handle.remove()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    output_data = {
        "metadata": {
            "model": args.model_name or args.model, "layer": args.layer,
            "feature": args.feature, "multiplier": args.multiplier,
            "max_tokens": args.max_tokens, "temperature": args.temperature,
            "pre_down_proj": args.pre_down_proj, "additive": additive,
            "architecture": args.architecture, "config": args.config,
            "harmbench_191_filtered": (not args.full_harmbench and args.config == "standard"),
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
