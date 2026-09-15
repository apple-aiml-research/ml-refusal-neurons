#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Evaluate LlamaGuard as a harmful-prompt classifier baseline.

Supports the same four datasets as evaluate_detection.py via --dataset:
  xstest            walledai/XSTest (test split)
  wildguard         allenai/wildguardmix wildguardtest (test split)
  toxicchat         lmsys/toxic-chat toxicchat0124 (test split)
  openai_moderation mmathys/openai-moderation-api-evaluation (1680 examples)

For each prompt, runs LlamaGuard to get a safe/unsafe verdict. AUROC/AUPRC are
threshold-independent (computed over the continuous unsafe-token probability);
accuracy/F1/precision/recall are computed at LlamaGuard's own native decision
(the "safe"/"unsafe" token it actually generates) — matching the paper's
Table 2 methodology, not a post-hoc optimal threshold. For a threshold-swept
comparison against neuron-activation-based discrimination, see
evaluate_llamaguard2.py instead.

Usage:
    python detection/evaluate_llamaguard.py \
        --guard_model meta-llama/Llama-Guard-3-8B --dataset xstest
"""

import argparse
import json
import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate_detection import LOADERS


def classify_prompt(model, tokenizer, prompt, device):
    """Returns (is_unsafe, unsafe_score, response_text)."""
    chat = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt")
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    input_ids = input_ids.to(device)

    newline_id = tokenizer.encode("\n\n", add_special_tokens=False)[0]
    input_with_newline = torch.cat(
        [input_ids, torch.tensor([[newline_id]], device=device)], dim=1)

    with torch.no_grad():
        outputs = model(input_with_newline)
        probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)

    safe_ids = tokenizer.encode("safe", add_special_tokens=False)
    unsafe_ids = tokenizer.encode("unsafe", add_special_tokens=False)
    safe_prob = probs[safe_ids[0]].item() if safe_ids else 0
    unsafe_prob = probs[unsafe_ids[0]].item() if unsafe_ids else 0
    total = safe_prob + unsafe_prob
    unsafe_score = unsafe_prob / total if total > 0 else 0.5

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids, max_new_tokens=100,
            do_sample=False, pad_token_id=0)
    response = tokenizer.decode(
        output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

    is_unsafe = "unsafe" in response.lower()
    return is_unsafe, unsafe_score, response


def compute_metrics(harmful_binary, harmless_binary, harmful_scores, harmless_scores):
    """
    AUROC/AUPRC from the continuous unsafe-token probability (threshold-independent).
    Accuracy/F1/precision/recall from LlamaGuard's own native safe/unsafe decision.
    """
    h_score = np.array(harmful_scores, dtype=float)
    s_score = np.array(harmless_scores, dtype=float)

    labels = np.concatenate([np.ones(len(h_score)), np.zeros(len(s_score))])
    scores = np.concatenate([h_score, s_score])

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    sorted_labels = labels[np.argsort(scores)]
    tp = n_pos
    auroc = 0.0
    for lbl in sorted_labels:
        if lbl == 1:
            tp -= 1
        else:
            auroc += tp
    auroc = auroc / (n_pos * n_neg) if (n_pos * n_neg) > 0 else 0.5

    desc_labels = labels[np.argsort(-scores)]
    tp_cum = np.cumsum(desc_labels)
    n_pred = np.arange(1, len(desc_labels) + 1)
    precision_curve = tp_cum / n_pred
    recall_curve = tp_cum / n_pos if n_pos > 0 else tp_cum
    auprc = sum(
        (recall_curve[i] - recall_curve[i - 1]) * precision_curve[i]
        for i in range(1, len(recall_curve)) if recall_curve[i] != recall_curve[i - 1]
    )

    h_bin = np.array(harmful_binary, dtype=int)
    s_bin = np.array(harmless_binary, dtype=int)
    tp_count = int(h_bin.sum())
    tn_count = int((1 - s_bin).sum())
    fp_count = int(s_bin.sum())
    fn_count = int(len(h_bin) - h_bin.sum())
    accuracy = (tp_count + tn_count) / (len(h_bin) + len(s_bin))
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "harmful_detected": tp_count,
        "harmful_total": len(h_bin),
        "harmful_detection_rate": tp_count / len(h_bin) if len(h_bin) > 0 else 0,
        "harmless_safe": tn_count,
        "harmless_total": len(s_bin),
        "harmless_safe_rate": tn_count / len(s_bin) if len(s_bin) > 0 else 0,
        "false_positive_rate": fp_count / len(s_bin) if len(s_bin) > 0 else 0,
        "false_negative_rate": fn_count / len(h_bin) if len(h_bin) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="LlamaGuard classifier baseline evaluation")
    parser.add_argument("--guard_model", type=str, default="meta-llama/Llama-Guard-3-8B")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["xstest", "wildguard", "toxicchat", "openai_moderation"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap prompts per class (for debugging)")
    args = parser.parse_args()

    print(f"Loading {args.dataset}...")
    loader = LOADERS[args.dataset]
    if args.dataset in ("xstest", "openai_moderation"):
        harmful, harmless = loader()
    else:
        harmful, harmless = loader(args.split)
    if args.limit:
        harmful = harmful[:args.limit]
        harmless = harmless[:args.limit]
    print(f"Loaded {len(harmful)} harmful + {len(harmless)} harmless")

    print(f"\nLoading guard model: {args.guard_model} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
             else torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.guard_model, dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.guard_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded.\n")

    harmful_binary, harmful_scores, harmful_details = [], [], []
    for prompt in tqdm(harmful, desc="Harmful"):
        is_unsafe, score, response = classify_prompt(model, tokenizer, prompt, device)
        harmful_binary.append(int(is_unsafe))
        harmful_scores.append(score)
        harmful_details.append({"prompt": prompt, "predicted_unsafe": is_unsafe,
                                 "unsafe_score": score, "response": response})

    harmless_binary, harmless_scores, harmless_details = [], [], []
    for prompt in tqdm(harmless, desc="Harmless"):
        is_unsafe, score, response = classify_prompt(model, tokenizer, prompt, device)
        harmless_binary.append(int(is_unsafe))
        harmless_scores.append(score)
        harmless_details.append({"prompt": prompt, "predicted_unsafe": is_unsafe,
                                  "unsafe_score": score, "response": response})

    metrics = compute_metrics(harmful_binary, harmless_binary, harmful_scores, harmless_scores)

    print("\n" + "=" * 60)
    print(f"LLAMAGUARD {args.dataset.upper()}")
    print("=" * 60)
    print(f"AUROC:               {metrics['auroc']:.4f}")
    print(f"AUPRC:               {metrics['auprc']:.4f}")
    print(f"Accuracy:            {metrics['accuracy']:.1%}")
    print(f"F1:                  {metrics['f1']:.4f}")
    print(f"Precision:           {metrics['precision']:.4f}")
    print(f"Recall:              {metrics['recall']:.4f}")
    print(f"Harmful detected:    {metrics['harmful_detected']}/{metrics['harmful_total']} "
          f"({metrics['harmful_detection_rate']:.1%})")
    print(f"Harmless safe:       {metrics['harmless_safe']}/{metrics['harmless_total']} "
          f"({metrics['harmless_safe_rate']:.1%})")
    print(f"False positive rate: {metrics['false_positive_rate']:.1%}")
    print(f"False negative rate: {metrics['false_negative_rate']:.1%}")

    fp = [d for d in harmless_details if d['predicted_unsafe']]
    fn = [d for d in harmful_details if not d['predicted_unsafe']]
    if fp:
        print(f"\nTop false positives ({len(fp)} total):")
        for d in fp[:5]:
            print(f"  {d['prompt'][:80]}")
    if fn:
        print(f"\nTop false negatives ({len(fn)} total):")
        for d in fn[:5]:
            print(f"  {d['prompt'][:80]}")

    if args.output is None:
        args.output = os.path.join(REPO_ROOT, "results", "detection", args.dataset,
                                    f"llamaguard_{args.dataset}.json")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results = {
        "metadata": {
            "guard_model": args.guard_model,
            "dataset": args.dataset,
            "split": args.split,
            "timestamp": datetime.now().isoformat(),
            "n_harmful": len(harmful),
            "n_harmless": len(harmless),
        },
        "metrics": metrics,
        "harmful_scores": harmful_scores,
        "harmless_scores": harmless_scores,
        "harmful_details": harmful_details,
        "harmless_details": harmless_details,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
