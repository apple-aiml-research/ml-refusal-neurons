#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
build_train_val.py — Build train/val prompt files for refusal neuron discovery.

Downloads all source data from public URLs / HuggingFace (no local files needed).

Harmful train pool: AdvBench + MaliciousInstruct + TDC2023, deduplicated against
  HarmBench standard (our val) and JBB-Behaviors (our test).

Val: HarmBench standard (walledai/HarmBench) minus JBB overlap = ~191 prompts.

Harmless: Alpaca instructions (tatsu-lab/alpaca), 60/20/20 split (seed=42),
  train portion sampled at --seed.

Outputs (written to data/splits/ by default):
  harmful_train.json   flat list of 128 items  (seed-dependent)
  harmless_train.json  flat list of 128 items  (seed-dependent)
  harmful_val.json     flat list of ~191 items (same for all seeds)
  harmless_val.json    flat list of 32 items   (seed-dependent)
  harmful_test.json    flat list of 100 items  (JBB, same for all seeds)
  harmless_test.json   flat list of 100 items  (first 100 of train pool not in harmless_train; guaranteed 0 overlap)

Usage:
    python find_refusal_neurons/build_train_val.py --seed 42
    python find_refusal_neurons/build_train_val.py --seed 123
"""

import argparse
import json
import os
import random
import sys

import requests


def download_text(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def download_advbench():
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    import csv, io
    text = download_text(url)
    reader = csv.DictReader(io.StringIO(text))
    return [row["goal"].strip() for row in reader if row.get("goal", "").strip()]


def download_malicious_instruct():
    url = "https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt"
    return [line.strip() for line in download_text(url).splitlines() if line.strip()]


def download_tdc2023():
    urls = [
        "https://raw.githubusercontent.com/centerforaisafety/tdc2023-starter-kit/main/red_teaming/data/dev/behaviors.json",
        "https://raw.githubusercontent.com/centerforaisafety/tdc2023-starter-kit/main/red_teaming/data/test/behaviors.json",
    ]
    items = []
    for url in urls:
        items.extend(json.loads(download_text(url)))
    return [s.strip() for s in items if isinstance(s, str) and s.strip()]


def load_harmbench_prompts():
    from datasets import load_dataset
    ds = load_dataset("walledai/HarmBench", "standard", split="train")
    field = "Behavior" if "Behavior" in ds.column_names else "prompt"
    return [row[field].strip() for row in ds]


def load_jbb_prompts():
    from datasets import load_dataset
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    return [row["Goal"].strip() for row in ds]


def load_alpaca():
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    return [row["instruction"].strip() for row in ds
            if row.get("input", "").strip() == "" and row.get("instruction", "").strip()]


def sample(pool, n, seed):
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def to_entries(texts, source):
    return [{"text": t, "source": source} for t in texts]


def main():
    parser = argparse.ArgumentParser(
        description="Build train/val prompts from public sources (no local files needed)"
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling train split (default: 42)")
    parser.add_argument("--n_train", type=int, default=128)
    parser.add_argument("--n_harmless_val", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "..", "data", "splits")
    os.makedirs(output_dir, exist_ok=True)

    # ── Download harmful sources ───────────────────────────────────────────────
    print("Downloading harmful sources...")
    advbench    = download_advbench()
    malicious   = download_malicious_instruct()
    tdc2023     = download_tdc2023()
    print(f"  AdvBench: {len(advbench)}, MaliciousInstruct: {len(malicious)}, TDC2023: {len(tdc2023)}")

    # ── Download val (HarmBench) and test (JBB) for deduplication ─────────────
    print("\nDownloading HarmBench standard (val) and JBB (test)...")
    hb_prompts = load_harmbench_prompts()
    jbb_prompts = load_jbb_prompts()
    print(f"  HarmBench: {len(hb_prompts)}, JBB: {len(jbb_prompts)}")

    # ── Build harmful train pool, deduplicated against val + test ─────────────
    exclude = set(hb_prompts) | set(jbb_prompts)
    pool = []
    seen = set()
    for text in advbench + malicious + tdc2023:
        if text not in exclude and text not in seen:
            pool.append(text)
            seen.add(text)
    print(f"\n  Harmful train pool after dedup: {len(pool)} prompts "
          f"(removed {len(advbench) + len(malicious) + len(tdc2023) - len(pool)})")

    # ── Build val: HarmBench minus JBB ────────────────────────────────────────
    jbb_set = set(jbb_prompts)
    harmful_val = [p for p in hb_prompts if p not in jbb_set]
    print(f"  Val (HarmBench minus JBB): {len(harmful_val)} prompts")

    # ── Download harmless (Alpaca) ─────────────────────────────────────────────
    print("\nDownloading Alpaca (harmless)...")
    alpaca = load_alpaca()
    print(f"  Alpaca instructions (no input): {len(alpaca)}")

    # 60/20/20 split with seed=42 matching Arditi
    rng42 = random.Random(42)
    alpaca_shuffled = list(alpaca)
    rng42.shuffle(alpaca_shuffled)
    train_size = int(0.6 * len(alpaca_shuffled))
    val_size   = int(0.2 * len(alpaca_shuffled))
    harmless_train_pool = alpaca_shuffled[:train_size]
    harmless_val_pool   = alpaca_shuffled[train_size:train_size + val_size]
    print(f"  Harmless train pool: {len(harmless_train_pool)}, val pool: {len(harmless_val_pool)}")

    # ── Write val/test files (same for all seeds) ─────────────────────────────
    harmful_val_path  = os.path.join(output_dir, "harmful_val.json")
    harmful_test_path = os.path.join(output_dir, "harmful_test.json")
    with open(harmful_val_path, "w") as f:
        json.dump(to_entries(harmful_val, "harmbench_standard"), f, indent=2)
    with open(harmful_test_path, "w") as f:
        json.dump(to_entries(jbb_prompts, "jbb"), f, indent=2)
    print(f"\n  -> {harmful_val_path}  ({len(harmful_val)} prompts)")
    print(f"  -> {harmful_test_path}  ({len(jbb_prompts)} prompts)")

    # ── Generate train split ───────────────────────────────────────────────────
    seed = args.seed
    harmful_train  = sample(pool,                args.n_train,        seed=seed)
    harmless_train = sample(harmless_train_pool, args.n_train,        seed=seed)
    harmless_val   = sample(harmless_val_pool,   args.n_harmless_val, seed=seed)

    # harmless_test: first 100 of train pool that are not in harmless_train
    harmless_train_set = set(harmless_train)
    harmless_test = []
    for t in harmless_train_pool:
        if t not in harmless_train_set:
            harmless_test.append(t)
            if len(harmless_test) == 100:
                break

    harmful_path       = os.path.join(output_dir, "harmful_train.json")
    harmless_path      = os.path.join(output_dir, "harmless_train.json")
    harmless_val_path  = os.path.join(output_dir, "harmless_val.json")
    harmless_test_path = os.path.join(output_dir, "harmless_test.json")

    with open(harmful_path, "w") as f:
        json.dump(to_entries(harmful_train, "arditi_pool"), f, indent=2)
    with open(harmless_path, "w") as f:
        json.dump(to_entries(harmless_train, "alpaca"), f, indent=2)
    with open(harmless_val_path, "w") as f:
        json.dump(to_entries(harmless_val, "alpaca"), f, indent=2)
    with open(harmless_test_path, "w") as f:
        json.dump(to_entries(harmless_test, "alpaca"), f, indent=2)

    print(f"\n  -> {harmful_path}  ({len(harmful_train)} prompts)")
    print(f"  -> {harmless_path}  ({len(harmless_train)} prompts)")
    print(f"  -> {harmless_val_path}  ({len(harmless_val)} prompts)")
    print(f"  -> {harmless_test_path}  ({len(harmless_test)} prompts, 0 overlap with harmless_train guaranteed)")

    print("\nDone. Example usage:")
    print(f"  --harmful_path  data/splits/harmful_train.json \\")
    print(f"  --harmless_path data/splits/harmless_train.json")


if __name__ == "__main__":
    main()
