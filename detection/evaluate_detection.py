#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Evaluate refusal neuron activation as a harmful-prompt discriminator.

Supports four datasets via --dataset:
  xstest            walledai/XSTest (test split)
  wildguard         allenai/wildguardmix wildguardtest (test split)
  toxicchat         lmsys/toxic-chat toxicchat0124 (test split)
  openai_moderation mmathys/openai-moderation-api-evaluation (1680 examples)

Usage:
    python detection/evaluate_detection.py \
        --model Qwen/Qwen3-14B --layer 17 --feature 2154 \
        --token_pos="-5,-6,-7,-8,-9" --agg min \
        --dataset xstest

    python detection/evaluate_detection.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct --layer 11 --feature 4258 \
        --token_pos="-2,-3,-4,-5" --agg max \
        --dataset wildguard

    python detection/evaluate_detection.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct --layer 11 --feature 4258 \
        --token_pos="-2,-3,-4,-5" --agg max \
        --dataset toxicchat
"""

import argparse
import json
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def detect_architecture(model):
    for name, _ in model.named_modules():
        if 'gate_proj' in name:
            return 'llama'
    for name, module in model.named_modules():
        if 'mlp' in name.lower() and hasattr(module, 'c_fc'):
            return 'gpt2'
    return 'unknown'


# ---------------------------------------------------------------------------
# Capture hook
# ---------------------------------------------------------------------------

def capture_activation(model, layer_num, feature_idx, input_ids):
    captured = {}

    def hook_fn(module, inp):
        features = inp[0]
        captured['features'] = features[:, :, feature_idx].detach().clone()

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
# Data loading
# ---------------------------------------------------------------------------

def load_xstest():
    from datasets import load_dataset
    ds = load_dataset("walledai/XSTest", split="test")
    harmful, harmless = [], []
    for ex in ds:
        label = ex.get("label", ex.get("type", ""))
        prompt = ex.get("prompt", ex.get("text", ""))
        if not prompt:
            continue
        if label in ("unsafe", "harmful", 1, "1"):
            harmful.append(prompt)
        else:
            harmless.append(prompt)
    return harmful, harmless


def load_wildguard(split="test"):
    from datasets import load_dataset
    config = "wildguardtrain" if split == "train" else "wildguardtest"
    hf_split = "train" if split == "train" else "test"
    ds = load_dataset("allenai/wildguardmix", config, split=hf_split)
    harmful, harmless = [], []
    for ex in ds:
        label = ex.get("prompt_harm_label", "")
        prompt = ex.get("prompt", "")
        if not prompt:
            continue
        if label == "harmful":
            harmful.append(prompt)
        elif label == "unharmful":
            harmless.append(prompt)
    if not harmful and not harmless:
        raise ValueError(
            f"No prompts loaded. Column names: {ds.column_names}. "
            f"Expected 'prompt' and 'prompt_harm_label'.")
    return harmful, harmless


def load_toxicchat(split="test"):
    from datasets import load_dataset
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split=split)
    harmful, harmless = [], []
    for ex in ds:
        prompt = ex.get("user_input", "")
        if not prompt:
            continue
        if ex.get("toxicity") == 1:
            harmful.append(prompt)
        else:
            harmless.append(prompt)
    return harmful, harmless


def load_openai_moderation(split="train"):
    from datasets import load_dataset
    ds = load_dataset("mmathys/openai-moderation-api-evaluation", split="train")
    _HARM_COLS = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]
    harmful, harmless = [], []
    for ex in ds:
        prompt = ex.get("prompt", "")
        if not prompt:
            continue
        if any(ex.get(col) == 1 for col in _HARM_COLS):
            harmful.append(prompt)
        else:
            harmless.append(prompt)
    return harmful, harmless


LOADERS = {
    "xstest": load_xstest,
    "wildguard": load_wildguard,
    "toxicchat": load_toxicchat,
    "openai_moderation": load_openai_moderation,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def tpr_at_fpr(h, s, fpr_target):
    """Maximum TPR achievable while keeping FPR <= fpr_target."""
    thresholds = np.sort(np.unique(np.concatenate([h, s])))[::-1]
    best_tpr = 0.0
    for t in thresholds:
        fpr = np.sum(s >= t) / len(s) if len(s) > 0 else 0
        if fpr > fpr_target:
            break
        best_tpr = np.sum(h >= t) / len(h) if len(h) > 0 else 0
    return float(best_tpr)


def compute_metrics(harmful_vals, harmless_vals, higher_is_harmful=True):
    h = np.array(harmful_vals)
    s = np.array(harmless_vals)

    if not higher_is_harmful:
        h, s = -h, -s

    labels = np.concatenate([np.ones(len(h)), np.zeros(len(s))])
    scores = np.concatenate([h, s])

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
    precision = tp_cum / n_pred
    recall = tp_cum / n_pos if n_pos > 0 else tp_cum
    auprc = sum(
        (recall[i] - recall[i - 1]) * precision[i]
        for i in range(1, len(recall)) if recall[i] != recall[i - 1]
    )

    # Best accuracy threshold (for FN/FP reporting)
    all_vals = np.sort(np.unique(scores))
    best_acc, best_thresh = 0, 0
    for thresh in all_vals:
        acc_gt = (np.sum(h > thresh) + np.sum(s <= thresh)) / (len(h) + len(s))
        if acc_gt > best_acc:
            best_acc, best_thresh = acc_gt, thresh

    # Optimal F1 threshold (sweep independently)
    best_f1, best_f1_thresh, best_prec, best_rec = 0, 0, 0, 0
    for thresh in all_vals:
        tp_count = int(np.sum(h > thresh))
        fp_count = int(np.sum(s > thresh))
        fn_count = int(np.sum(h <= thresh))
        prec = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        rec = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_f1_thresh = f1, thresh
            best_prec, best_rec = prec, rec

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "best_accuracy": float(best_acc),
        "best_threshold": float(best_thresh),
        "best_direction": ">",
        "f1": float(best_f1),
        "precision": float(best_prec),
        "recall": float(best_rec),
        "f1_threshold": float(best_f1_thresh),
        "tpr_at_fpr1": tpr_at_fpr(h, s, 0.01),
        "tpr_at_fpr2": tpr_at_fpr(h, s, 0.02),
        "tpr_at_fpr3": tpr_at_fpr(h, s, 0.03),
        "tpr_at_fpr5": tpr_at_fpr(h, s, 0.05),
        "harmful_mean": float(h.mean()),
        "harmful_std": float(h.std()),
        "harmful_median": float(np.median(h)),
        "harmless_mean": float(s.mean()),
        "harmless_std": float(s.std()),
        "harmless_median": float(np.median(s)),
        "n_harmful": int(len(h)),
        "n_harmless": int(len(s)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate refusal neuron discriminability")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--feature", type=int, required=True)
    parser.add_argument("--token_pos", type=str, required=True,
                        help="Token position(s) from end. Single: -6. Multiple: -5,-6,-7,-8,-9")
    parser.add_argument("--agg", type=str, default=None, choices=["min", "max"],
                        help="Aggregation: min for positive best_mult, max for negative best_mult")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["xstest", "wildguard", "toxicchat", "openai_moderation"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap prompts per class")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    token_positions = [int(x.strip()) for x in args.token_pos.split(",")]

    print("=" * 70)
    print("NEURON DISCRIMINATION EVALUATION")
    print("=" * 70)
    print(f"Model:    {args.model}")
    print(f"Layer:    {args.layer}   Feature: {args.feature}")
    print(f"token_pos: {args.token_pos}  agg: {args.agg}")
    print(f"Dataset:  {args.dataset}  split: {args.split}"
          + (f"  limit: {args.limit}/class" if args.limit else ""))
    print("=" * 70)

    print(f"Loading {args.dataset}...")
    loader = LOADERS[args.dataset]
    if args.dataset in ("xstest", "openai_moderation"):
        harmful_prompts, harmless_prompts = loader()
    else:
        harmful_prompts, harmless_prompts = loader(args.split)
    if args.limit:
        harmful_prompts = harmful_prompts[:args.limit]
        harmless_prompts = harmless_prompts[:args.limit]
    print(f"Harmful: {len(harmful_prompts)}  Harmless: {len(harmless_prompts)}")

    print(f"\nLoading model: {args.model} ...")
    dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
             else (torch.float16 if torch.cuda.is_available() else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    ).eval()
    model.generation_config.max_length = None
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"Architecture: {detect_architecture(model)}\n")

    def _build_text(prompt):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    def get_activations(prompts, label):
        vals = []
        for prompt in tqdm(prompts, desc=label):
            text = _build_text(prompt)
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
            all_act = capture_activation(model, args.layer, args.feature, input_ids)
            sl = input_ids.shape[1]
            pos_vals = []
            for tp in token_positions:
                ap = tp if tp >= 0 else sl + tp
                if 0 <= ap < sl:
                    pos_vals.append(all_act[0, ap].item())
            if args.agg == "min" and pos_vals:
                vals.append(min(pos_vals))
            elif args.agg == "max" and pos_vals:
                vals.append(max(pos_vals))
            elif pos_vals:
                vals.append(pos_vals[0])
            else:
                vals.append(0.0)
        return vals

    harmful_vals = get_activations(harmful_prompts, "Harmful")
    harmless_vals = get_activations(harmless_prompts, "Harmless")

    metrics = compute_metrics(harmful_vals, harmless_vals, higher_is_harmful=(args.agg == "max"))

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Harmful  (n={metrics['n_harmful']:4d}):  "
          f"mean={metrics['harmful_mean']:+.4f}  std={metrics['harmful_std']:.4f}  "
          f"median={metrics['harmful_median']:+.4f}")
    print(f"Harmless (n={metrics['n_harmless']:4d}):  "
          f"mean={metrics['harmless_mean']:+.4f}  std={metrics['harmless_std']:.4f}  "
          f"median={metrics['harmless_median']:+.4f}")
    print()
    print(f"AUROC:          {metrics['auroc']:.4f}")
    print(f"AUPRC:          {metrics['auprc']:.4f}")
    print(f"Optimal F1:     {metrics['f1']:.4f}")
    print(f"Best accuracy:  {metrics['best_accuracy']:.1%}")
    print(f"TPR@FPR 1%:     {metrics['tpr_at_fpr1']:.1%}")
    print(f"TPR@FPR 2%:     {metrics['tpr_at_fpr2']:.1%}")
    print(f"TPR@FPR 3%:     {metrics['tpr_at_fpr3']:.1%}")
    print(f"TPR@FPR 5%:     {metrics['tpr_at_fpr5']:.1%}")

    thresh = metrics['best_threshold']
    fn = sorted([(v, p) for v, p in zip(harmful_vals, harmful_prompts) if v <= thresh],
                key=lambda x: x[0])
    fp = sorted([(v, p) for v, p in zip(harmless_vals, harmless_prompts) if v > thresh],
                key=lambda x: -x[0])
    if fn:
        print(f"\nTop 5 false negatives ({len(fn)} total):")
        for v, p in fn[:5]:
            print(f"  act={v:+.4f}  {p[:90]}")
    if fp:
        print(f"\nTop 5 false positives ({len(fp)} total):")
        for v, p in fp[:5]:
            print(f"  act={v:+.4f}  {p[:90]}")

    if args.output is None:
        model_name = args.model.split("/")[-1]
        agg_str = f"_agg{args.agg}" if args.agg else f"_TP{args.token_pos}"
        limit_str = f"_n{args.limit}" if args.limit else ""
        args.output = os.path.join(
            REPO_ROOT, "results", "detection", args.dataset,
            f"{model_name}_L{args.layer}_F{args.feature}{agg_str}{limit_str}.json")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results = {
        "metadata": {
            "model": args.model,
            "layer": args.layer,
            "feature": args.feature,
            "token_pos": args.token_pos,
            "agg": args.agg,
            "dataset": args.dataset,
            "split": args.split,
            "timestamp": datetime.now().isoformat(),
        },
        "metrics": metrics,
        "harmful_activations": harmful_vals,
        "harmless_activations": harmless_vals,
        "harmful_prompts": harmful_prompts,
        "harmless_prompts": harmless_prompts,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to: {args.output}")

    try:
        import matplotlib
        from tueplots import bundles
        bundle = bundles.neurips2024()
        bundle["text.usetex"] = False
        bundle["figure.figsize"] = (1.32, 1.6)
        bundle["axes.spines.top"] = True
        bundle["axes.spines.right"] = True
        matplotlib.rcParams.update(bundle)
        matplotlib.rcParams.update({
            "savefig.dpi": 300, "figure.dpi": 300,
            "figure.facecolor": "white", "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#888888", "axes.linewidth": 0.6,
        })
        import matplotlib.pyplot as plt

        h = np.array(harmful_vals)
        s = np.array(harmless_vals)
        scores_all = np.concatenate([h, s])
        thresholds = np.sort(np.unique(scores_all))
        tprs, fprs = [], []
        for t in thresholds:
            tprs.append(np.sum(h > t) / len(h))
            fprs.append(np.sum(s > t) / len(s))

        fig, ax = plt.subplots()
        ax.plot(fprs, tprs, color="#CC3311", linewidth=1.2, zorder=3)
        ax.plot([0, 1], [0, 1], color="#BBBBBB", linewidth=0.6, linestyle="--", zorder=1)
        model_short = args.model.split("/")[-1]
        ax.set_title(f"{model_short}, L{args.layer}:F{args.feature}", fontsize=7, pad=3)
        ax.set_xlabel("FPR", fontsize=7)
        ax.set_ylabel("TPR", fontsize=7)
        ax.tick_params(axis="both", length=2.5, labelsize=6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.text(0.97, 0.05, f"AUROC={metrics['auroc']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="#CC3311")
        for spine in ax.spines.values():
            spine.set_color("#888888")
            spine.set_linewidth(0.6)
        plot_path = os.path.splitext(args.output)[0] + "_roc"
        plt.tight_layout(pad=0.4)
        plt.savefig(plot_path + ".pdf", bbox_inches='tight', pad_inches=0.02)
        plt.savefig(plot_path + ".png", bbox_inches='tight', pad_inches=0.02)
        plt.close()
        print(f"ROC plot: {plot_path}.{{pdf,png}}")
    except ImportError as e:
        print(f"Skipping ROC plot (missing dependency: {e})")

    print("=" * 70)


if __name__ == "__main__":
    main()
