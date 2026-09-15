#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Find Features for a Concept — with Activation Gap

Discovers MLP features that encode a specific semantic concept by ranking
each neuron on cross-validated logistic-regression accuracy over per-example
activations of positive vs. negative passages. Also reports activation-magnitude
activation_gap metrics (h_act, n_act, activation_gap_q1, activation_gap_median, combined_q1,
combined_median) to help disambiguate among tied high-CV features, and prints
per-metric rankings plus an average rank across the user-selected metric set
(see --rankings).
"""

import argparse
import json
import sys
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Add parent directory to path
sys.path.insert(0, SCRIPT_DIR)

from discovery import FeatureDiscovery


# ANSI colors + block chars for terminal sparklines
_SPARK_BLOCKS = " ▁▂▃▄▅▆▇█"
_GREEN = "\033[92m"
_RED   = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def _sparkline_single(counts, max_count, color):
    """Render counts as a colored unicode sparkline scaled to max_count.
    Non-zero counts always render as at least ▁ (never rounded down to space)."""
    if max_count <= 0:
        return color + _SPARK_BLOCKS[0] * len(counts) + _RESET
    n_levels = len(_SPARK_BLOCKS) - 1
    out = []
    for c in counts:
        if c <= 0:
            out.append(_SPARK_BLOCKS[0])
        else:
            idx = min(n_levels, max(1, int(round(c / max_count * n_levels))))
            out.append(_SPARK_BLOCKS[idx])
    return color + "".join(out) + _RESET


def _sparkline_combined(pos_counts, neg_counts, p_max, n_max):
    """Per-bin overlay: color = winner (green pos, red neg, yellow tie);
    height = winner_count / winner_class_peak. Scaling is per-class so a huge
    spike in one class doesn't flatten the other class's shape to invisibility."""
    out = []
    n_levels = len(_SPARK_BLOCKS) - 1
    for p, n in zip(pos_counts, neg_counts):
        if p == 0 and n == 0:
            out.append(_SPARK_BLOCKS[0])
            continue
        if p > n * 1.2:
            color = _GREEN
            frac = (p / p_max) if p_max > 0 else 0.0
        elif n > p * 1.2:
            color = _RED
            frac = (n / n_max) if n_max > 0 else 0.0
        else:
            color = _YELLOW
            # tie: take the larger fractional height between the two per-class scales
            p_frac = (p / p_max) if p_max > 0 else 0.0
            n_frac = (n / n_max) if n_max > 0 else 0.0
            frac = max(p_frac, n_frac)
        idx = min(n_levels, max(1, int(round(frac * n_levels))))  # never round non-zero counts to space
        out.append(color + _SPARK_BLOCKS[idx] + _RESET)
    return "".join(out)


def print_activation_dist(rank, feat, pos_acts, neg_acts, n_bins=30, marker=""):
    """Print a 3-line colored activation-distribution block for one neuron.

    pos_acts / neg_acts: numpy arrays shaped (n_examples, n_features) for the
    layer that feat lives on. Reads the column feat['feature_idx']."""
    layer = feat['layer_idx']
    fi = feat['feature_idx']
    p = pos_acts[:, fi]
    n = neg_acts[:, fi]
    lo = float(min(p.min(), n.min()))
    hi = float(max(p.max(), n.max()))
    if hi - lo < 1e-9:
        hi = lo + 1e-6
    bins = np.linspace(lo, hi, n_bins + 1)
    p_counts, _ = np.histogram(p, bins=bins)
    n_counts, _ = np.histogram(n, bins=bins)
    p_max = int(p_counts.max()) if p_counts.size else 0
    n_max = int(n_counts.max()) if n_counts.size else 0

    header = f"{rank:3d}. Layer {layer:2d}, Feature {fi:5d} — bins={n_bins}, range=[{lo:+.2f}, {hi:+.2f}], N={len(p)}/{len(n)} | h={float(p.mean()):+.3f} n={float(n.mean()):+.3f}{marker}"
    print(f"  {header}")
    print(f"       pos: {_sparkline_single(p_counts, p_max, _GREEN)} (peak bin count = {p_max})")
    print(f"       neg: {_sparkline_single(n_counts, n_max, _RED)}   (peak bin count = {n_max})")
    print(f"       mix: {_sparkline_combined(p_counts, n_counts, p_max, n_max)} (green=pos-dominant, red=neg-dominant, yellow≈tied; per-class scale)")


def bimodality_ratio(scores):
    """Largest gap between consecutive sorted activations / total range.
    High = discrete clusters (word trigger). Low = graded/continuous (concept neuron)."""
    s = sorted(scores)
    total = s[-1] - s[0]
    if total < 1e-9:
        return 0.0
    max_gap = max(s[i+1] - s[i] for i in range(len(s) - 1))
    return max_gap / total


def bimodality_robust(scores, min_side=0.10):
    """Like bimodality_ratio but only counts gaps where both sides have >= min_side fraction of examples.
    Ignores tail outliers — a single extreme example won't make a feature look bimodal."""
    s = sorted(scores)
    n = len(s)
    total = s[-1] - s[0]
    if total < 1e-9:
        return 0.0
    min_count = max(1, int(n * min_side))
    gaps = [(s[i+1] - s[i]) for i in range(n-1)
            if (i + 1) >= min_count and (n - i - 1) >= min_count]
    if not gaps:
        return 0.0
    return max(gaps) / total


def main():
    parser = argparse.ArgumentParser(description="Find features representing specific concepts with activation_gap ranking")
    parser.add_argument("--model", type=str, required=True, help="Model name/path")
    parser.add_argument("--concept", type=str, required=True, help="Concept to find")
    parser.add_argument("--k", type=int, default=100, help="Number of examples per class")
    parser.add_argument("--examples_file", type=str, required=True, help="Path to examples JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")

    # Architecture
    parser.add_argument("--architecture", type=str, default="auto", choices=["auto", "llama", "gpt2"],
                        help="Model architecture")
    parser.add_argument("--capture_stage", type=str, default="default",
                        choices=["default", "up_proj", "gate_proj", "gated", "down_proj", "residual"],
                        help="Which FFN stage to capture features at (or 'residual' for layer output)")

    # Discovery parameters
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer numbers to analyze")
    parser.add_argument("--top_features", type=int, default=10, help="Number of top features per layer")
    parser.add_argument("--cv_folds", type=int, default=2, help="Cross-validation folds")
    parser.add_argument("--normalize", action='store_true', help="Normalize features")
    parser.add_argument("--token_aggregation", type=str, default="max", choices=["max", "mean", "min"])
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument("--rankings", nargs='+', default=['activation_gap', 'cv'],
                        choices=['cv', 'activation_gap', 'q1', 'median', 'combined_q1', 'combined_median'],
                        help="Metrics to include in avg-rank (in the order specified — first metric "
                             "is the tie-breaker on avg_rank). Default: activation_gap cv. "
                             "cv=CV accuracy, activation_gap=|mean(pos)−mean(neg)| activation gap, q1=|q1 activation_gap|, "
                             "median=|median activation_gap|, combined_q1=q1*(1-bimodality)*selectivity, "
                             "combined_median=median*(1-bimodality)*selectivity. "
                             "Example: --rankings activation_gap q1 combined_q1 combined_median")
    parser.add_argument("--target", type=str, default=None, help="Highlight a target feature in format L:F (e.g. 20:4256). Its rank is always reported, even when it falls below the rows shown in the tables.")
    parser.add_argument("--filter_neg_mag", action='store_true',
                        help="Drop candidates where |n_act| >= |h_act| (i.e. keep only neurons that fire more strongly on positives than on negatives). "
                             "Matches the magnitude criterion used for refusal-neuron selection in Section 2 of the paper.")
    parser.add_argument("--show_dist", type=int, default=0,
                        help="After the aggregate-ranking table, print colored ANSI activation-distribution histograms (green=positive, red=negative, combined) for the top-N neurons matching that table's ranking. Set to 0 to disable.")
    parser.add_argument("--dist_bins", type=int, default=30, help="Number of histogram bins used by --show_dist (default: 30).")
    args = parser.parse_args()

    show_rankings = args.rankings  # metrics to print sorted tables + use in avg rank

    print("="*60)
    print("FEATURE DISCOVERY (with activation_gap)")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Concept: {args.concept}")
    print(f"Examples per class: {args.k}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Token aggregation: {args.token_aggregation}")
    print(f"Capture stage: {args.capture_stage}")
    print()

    # Load examples
    print("Loading examples...")
    try:
        with open(args.examples_file) as f:
            _data = json.load(f)
        positive_examples = [e if isinstance(e, str) else e.get("text", str(e)) for e in _data["positive_examples"]]
        negative_examples = [e if isinstance(e, str) else e.get("text", str(e)) for e in _data["negative_examples"]]
        if args.k:
            positive_examples = positive_examples[:args.k]
            negative_examples = negative_examples[:args.k]
        print(f"✓ Loaded {len(positive_examples)} positive and {len(negative_examples)} negative examples")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return 1

    # Load model
    print(f"\nLoading model: {args.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using {'float16 on CUDA' if torch.cuda.is_available() else 'float32 on CPU'}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"✓ Model loaded on {device}")

    # Parse layers
    layers_to_analyze = None
    if args.layers:
        layers_to_analyze = [int(x.strip()) for x in args.layers.split(',')]
        print(f"Analyzing specific layers: {layers_to_analyze}")

    # Initialize feature discovery
    discovery = FeatureDiscovery(
        model=model,
        tokenizer=tokenizer,
        device=device,
        architecture=args.architecture,
        capture_stage=args.capture_stage,
        cv_folds=args.cv_folds,
        normalize=args.normalize,
        token_aggregation=args.token_aggregation,
        n_jobs=args.n_jobs
    )

    # Find features (CV accuracy ranking)
    results = discovery.find_features(
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        layers_to_analyze=layers_to_analyze,
        top_k=args.top_features
    )

    # Recompute activations to get h_act/n_act for activation_gap (hooks still registered)
    positive_activations = discovery.collect_activations(positive_examples, "positive", layers_to_analyze)
    negative_activations = discovery.collect_activations(negative_examples, "negative", layers_to_analyze)

    # Enrich each candidate with h_act, n_act, activation_gap, median activation_gap, bimodality
    for feat in results['global_top_candidates']:
        layer_idx = feat['layer_idx']
        fi = feat['feature_idx']
        if layer_idx in positive_activations and layer_idx in negative_activations:
            pos_acts = positive_activations[layer_idx]
            neg_acts = negative_activations[layer_idx]
            feat['h_act'] = float(pos_acts[:, fi].mean())
            feat['n_act'] = float(neg_acts[:, fi].mean())
            feat['activation_gap'] = feat['h_act'] - feat['n_act']
            feat['h_med'] = float(np.median(pos_acts[:, fi]))
            feat['n_med'] = float(np.median(neg_acts[:, fi]))
            feat['activation_gap_median'] = feat['h_med'] - feat['n_med']
            feat['h_q1'] = float(np.percentile(pos_acts[:, fi], 25))
            feat['n_q1'] = float(np.percentile(neg_acts[:, fi], 25))
            feat['activation_gap_q1'] = feat['h_q1'] - feat['n_q1']
            pos_scores = pos_acts[:, fi].tolist()
            feat['bimodality'] = bimodality_ratio(pos_scores)
            feat['bimodality_robust'] = bimodality_robust(pos_scores)
            # selectivity: fraction of positive activation that is concept-specific (not baseline)
            feat['selectivity'] = float(1.0 - abs(feat['n_act']) / abs(feat['h_act'])) if abs(feat['h_act']) > 1e-9 else 0.0
            feat['combined_q1']     = feat['activation_gap_q1']      * (1 - feat['bimodality_robust']) * max(0.0, feat['selectivity'])
            feat['combined_median'] = feat['activation_gap_median']  * (1 - feat['bimodality_robust']) * max(0.0, feat['selectivity'])
            feat['combined'] = feat['combined_q1']  # backwards compat alias

    # Magnitude filter: keep only neurons whose |h_act| > |n_act|
    # (mirrors the refusal-neuron selection criterion in Section 2 of the paper).
    if args.filter_neg_mag:
        before = len(results['global_top_candidates'])
        results['global_top_candidates'] = [
            f for f in results['global_top_candidates']
            if 'h_act' in f and abs(f['h_act']) > abs(f['n_act'])
        ]
        after = len(results['global_top_candidates'])
        print(f"\nMagnitude filter (|h_act| > |n_act|): kept {after}/{before} candidates.")

    # Save per-example activations for top-20 features by |activation_gap|
    top20_by_activation_gap = sorted(
        results['global_top_candidates'],
        key=lambda x: abs(x.get('activation_gap', 0)),
        reverse=True
    )[:20]

    per_example_records = []
    for feat in top20_by_activation_gap:
        layer_idx = feat['layer_idx']
        fi = feat['feature_idx']
        if layer_idx in positive_activations and layer_idx in negative_activations:
            pos_scores = positive_activations[layer_idx][:, fi].tolist()
            neg_scores = negative_activations[layer_idx][:, fi].tolist()
        else:
            pos_scores = []
            neg_scores = []
        per_example_records.append({
            "layer_idx": layer_idx,
            "feature_idx": fi,
            "cv_accuracy": feat['cv_accuracy'],
            "activation_gap": feat.get('activation_gap', None),
            "activation_gap_median": feat.get('activation_gap_median', None),
            "activation_gap_q1": feat.get('activation_gap_q1', None),
            "bimodality": feat.get('bimodality', None),
            "bimodality_robust": feat.get('bimodality_robust', None),
            "selectivity": feat.get('selectivity', None),
            "combined": feat.get('combined', None),
            "positive_scores": pos_scores,
            "negative_scores": neg_scores,
            "positive_examples": positive_examples[:len(pos_scores)],
            "negative_examples": negative_examples[:len(neg_scores)],
        })

    # Add metadata
    results["metadata"] = {
        "model": args.model,
        "concept": args.concept,
        "num_positive_examples": len(positive_examples),
        "num_negative_examples": len(negative_examples),
        "cv_folds": args.cv_folds,
        "normalize": args.normalize,
        "token_aggregation": args.token_aggregation,
        "capture_stage": args.capture_stage
    }

    # Save results
    if args.output is None:
        model_name = args.model.replace("/", "_")
        concept_name = args.concept.replace(" ", "_").replace("/", "_")
        args.output = os.path.join(REPO_ROOT, "results", "find_concept_neurons",
                                    f"features_{model_name}_{concept_name}.json")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    scores_output = args.output.replace(".json", "_scores.json")
    with open(scores_output, 'w') as f:
        json.dump(per_example_records, f, indent=2)

    print(f"\n✓ Results saved to {args.output}")
    print(f"✓ Per-example scores (top-20 by activation_gap) saved to {scores_output}")

    # Parse target feature if specified
    target_layer, target_feature = None, None
    target_feat = None
    if args.target:
        try:
            target_layer, target_feature = [int(x) for x in args.target.split(':')]
        except ValueError:
            print(f"Warning: --target '{args.target}' must be in format L:F (e.g. 20:4256), ignoring.")

    # Build target feat dict from activations (works even if not in global_top_candidates)
    if target_layer is not None:
        if target_layer in positive_activations and target_layer in negative_activations:
            pos_acts = positive_activations[target_layer]
            neg_acts = negative_activations[target_layer]
            fi = target_feature
            h_act = float(pos_acts[:, fi].mean())
            n_act = float(neg_acts[:, fi].mean())
            activation_gap = h_act - n_act
            h_med = float(np.median(pos_acts[:, fi]))
            n_med = float(np.median(neg_acts[:, fi]))
            h_q1 = float(np.percentile(pos_acts[:, fi], 25))
            n_q1 = float(np.percentile(neg_acts[:, fi], 25))
            target_feat = {
                'layer_idx': target_layer,
                'feature_idx': target_feature,
                'cv_accuracy': float('nan'),
                'h_act': h_act,
                'n_act': n_act,
                'activation_gap': activation_gap,
                'activation_gap_median': h_med - n_med,
                'activation_gap_q1': h_q1 - n_q1,
                'bimodality': bimodality_ratio(pos_acts[:, fi].tolist()),
                'bimodality_robust': bimodality_robust(pos_acts[:, fi].tolist()),
            }
            target_feat['selectivity'] = float(1.0 - abs(n_act) / abs(h_act)) if abs(h_act) > 1e-9 else 0.0
            target_feat['combined_q1']     = (h_q1 - n_q1)     * (1 - target_feat['bimodality_robust']) * max(0.0, target_feat['selectivity'])
            target_feat['combined_median'] = (h_med - n_med)   * (1 - target_feat['bimodality_robust']) * max(0.0, target_feat['selectivity'])
            target_feat['combined'] = target_feat['combined_q1']  # backwards compat alias
            # Check if it appears in global_top_candidates (CV already computed)
            for feat in results['global_top_candidates']:
                if feat['layer_idx'] == target_layer and feat['feature_idx'] == target_feature:
                    target_feat['cv_accuracy'] = feat['cv_accuracy']
                    break
            # If not in candidates, compute CV from scratch using logistic regression
            if np.isnan(target_feat['cv_accuracy']):
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import cross_val_score
                X = np.concatenate([pos_acts[:, fi:fi+1], neg_acts[:, fi:fi+1]], axis=0)
                y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))
                cv_scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=args.cv_folds)
                target_feat['cv_accuracy'] = float(cv_scores.mean())
        else:
            print(f"Warning: target layer {target_layer} not in analyzed layers, cannot compute stats.")

    def find_target_rank(candidates):
        """Return rank if target is in candidates, else compute hypothetical rank by score."""
        if target_layer is None or target_feat is None:
            return None
        for rank, feat in enumerate(candidates, 1):
            if feat['layer_idx'] == target_layer and feat['feature_idx'] == target_feature:
                return rank
        return None

    def hypothetical_rank(candidates, sort_key):
        """Where would target_feat rank if inserted into this sorted list?"""
        if target_feat is None:
            return None
        target_val = target_feat.get(sort_key, 0) or 0
        return sum(1 for f in candidates if (f.get(sort_key, 0) or 0) > target_val) + 1

    def print_target_if_needed(candidates, rank, sort_key=None):
        if target_feat is None:
            return
        if rank is None or rank > 23:
            hyp = hypothetical_rank(candidates, sort_key) if sort_key and rank is None else rank
            rank_str = f"rank {hyp} (hypothetical)" if rank is None else f"rank {rank}"
            print(f"\n  TARGET ({rank_str}):")
            print_feat_row(hyp or 0, target_feat, " ◄ TARGET")

    def print_feat_row(i, feat, marker=""):
        activation_gap = feat.get('activation_gap', float('nan'))
        med_activation_gap = feat.get('activation_gap_median', float('nan'))
        activation_gap_q1 = feat.get('activation_gap_q1', float('nan'))
        bimodality = feat.get('bimodality_robust', feat.get('bimodality', float('nan')))
        selectivity = feat.get('selectivity', float('nan'))
        combined = feat.get('combined', float('nan'))
        h_act = feat.get('h_act', float('nan'))
        n_act = feat.get('n_act', float('nan'))
        cv = feat.get('cv_accuracy', float('nan'))
        cv_str = f"{cv:.3f}" if cv == cv else "  N/A"
        print(f"{i:3d}. Layer {feat['layer_idx']:2d}, Feature {feat['feature_idx']:5d} | "
              f"CV: {cv_str} | h: {h_act:7.3f} | n: {n_act:7.3f} | "
              f"q1c: {activation_gap_q1:7.3f} | bim_r: {bimodality:.3f} | sel: {selectivity:.3f} | comb: {combined:7.3f}{marker}")

    # Map metric name -> (sort key string, key_fn, label)
    METRIC_CONFIGS = {
        'cv':             ('cv_accuracy',    lambda x: x.get('cv_accuracy', 0) or 0,              'CV accuracy'),
        'activation_gap':    ('activation_gap',       lambda x: abs(x.get('activation_gap', 0) or 0),             '|activation_gap|'),
        'q1':             ('activation_gap_q1',    lambda x: abs(x.get('activation_gap_q1', 0) or 0),          '|activation_gap_q1|'),
        'median':         ('activation_gap_median',lambda x: abs(x.get('activation_gap_median', 0) or 0),      '|activation_gap_median|'),
        'combined_q1':    ('combined_q1',    lambda x: x.get('combined_q1', 0) or 0,               'combined_q1'),
        'combined_median':('combined_median',lambda x: x.get('combined_median', 0) or 0,           'combined_median'),
    }

    # Use exactly the metrics the user specified, in the order they specified.
    # First metric is used as tie-breaker on avg_rank.
    active_metrics = list(show_rankings)

    # Print summary sorted by CV accuracy (always shown)
    print("\n" + "="*85)
    print(f"GLOBAL TOP FEATURES (sorted by CV accuracy)")
    print("="*85)
    for i, feat in enumerate(results['global_top_candidates'][:23], 1):
        marker = " ◄ TARGET" if (feat['layer_idx'] == target_layer and feat['feature_idx'] == target_feature) else ""
        print_feat_row(i, feat, marker)
    rank = find_target_rank(results['global_top_candidates'])
    print_target_if_needed(results['global_top_candidates'], rank, sort_key='cv_accuracy')

    # Print optional sorted tables
    for metric in show_rankings:
        if metric == 'cv':
            continue  # already printed above
        sort_key, key_fn, label = METRIC_CONFIGS[metric]
        sorted_cands = sorted(results['global_top_candidates'], key=key_fn, reverse=True)
        print("\n" + "="*85)
        print(f"GLOBAL TOP FEATURES (sorted by {label})")
        print("="*85)
        for i, feat in enumerate(sorted_cands[:23], 1):
            marker = " ◄ TARGET" if (feat['layer_idx'] == target_layer and feat['feature_idx'] == target_feature) else ""
            print_feat_row(i, feat, marker)
        rank = find_target_rank(sorted_cands)
        print_target_if_needed(sorted_cands, rank, sort_key=sort_key)

    # Avg-rank across active metrics
    candidates = results['global_top_candidates']

    def dense_ranks(feats, key_fn):
        sorted_vals = sorted([key_fn(f) for f in feats], reverse=True)
        val_to_rank = {}
        next_rank = 1
        for v in sorted_vals:
            if v not in val_to_rank:
                val_to_rank[v] = next_rank
                next_rank += 1
        return {id(f): val_to_rank[key_fn(f)] for f in feats}

    rank_maps = {m: dense_ranks(candidates, METRIC_CONFIGS[m][1]) for m in active_metrics}

    for feat in candidates:
        feat['avg_rank'] = sum(rank_maps[m][id(feat)] for m in active_metrics) / len(active_metrics)

    # Primary: avg_rank. Tie-break: rank in first metric of active_metrics, then second, etc.
    candidates_by_avg = sorted(
        candidates,
        key=lambda x: (
            x.get('avg_rank', 9999),
            *(rank_maps[m][id(x)] for m in active_metrics),
        ),
    )
    METRIC_ABBREV = {
        'cv': 'cv', 'activation_gap': 'gap', 'q1': 'q1',
        'median': 'med', 'combined_q1': 'cq1', 'combined_median': 'cmd',
    }

    metric_labels = ' / '.join(METRIC_CONFIGS[m][2] for m in active_metrics)

    print("\n" + "="*85)
    print(f"GLOBAL TOP FEATURES (sorted by average rank across {metric_labels})")
    print("="*85)
    for i, feat in enumerate(candidates_by_avg[:23], 1):
        marker = " ◄ TARGET" if (feat['layer_idx'] == target_layer and feat['feature_idx'] == target_feature) else ""
        ranks_str = ' '.join(f"{METRIC_ABBREV[m]}:{rank_maps[m][id(feat)]}" for m in active_metrics)
        vals_str = ' '.join(f"{METRIC_ABBREV[m]}={METRIC_CONFIGS[m][1](feat):7.3f}" for m in active_metrics)
        avg_r = feat['avg_rank']
        h_act = feat.get('h_act', float('nan'))
        n_act = feat.get('n_act', float('nan'))
        print(f"{i:3d}. Layer {feat['layer_idx']:2d}, Feature {feat['feature_idx']:5d} | "
              f"avg_rank: {avg_r:5.1f} ({ranks_str}) | {vals_str} | h: {h_act:7.3f} | n: {n_act:7.3f}{marker}")

    # Show target in avg-rank view if not in top 23
    if target_feat is not None:
        t_ranks = {m: hypothetical_rank(candidates, METRIC_CONFIGS[m][0]) for m in active_metrics}
        t_avg = sum(t_ranks.values()) / len(active_metrics)
        in_top = any(f['layer_idx'] == target_layer and f['feature_idx'] == target_feature
                     for f in candidates_by_avg[:23])
        if not in_top:
            hyp_avg = sum(1 for f in candidates_by_avg if f.get('avg_rank', 9999) < t_avg) + 1
            ranks_str = ' '.join(f"{METRIC_ABBREV[m]}:{t_ranks[m]}" for m in active_metrics)
            vals_str = ' '.join(f"{METRIC_ABBREV[m]}={METRIC_CONFIGS[m][1](target_feat):7.3f}" for m in active_metrics)
            t_h_act = target_feat.get('h_act', float('nan'))
            t_n_act = target_feat.get('n_act', float('nan'))
            print(f"\n  TARGET (rank {hyp_avg} hypothetical):")
            print(f"    Layer {target_layer:2d}, Feature {target_feature:5d} | "
                  f"avg_rank: {t_avg:5.1f} ({ranks_str}) | {vals_str} | h: {t_h_act:7.3f} | n: {t_n_act:7.3f} ◄ TARGET")

    # Optional: colored ANSI activation-distribution histograms for the top-N
    if args.show_dist and args.show_dist > 0:
        top_n = min(args.show_dist, len(candidates_by_avg))
        print("\n" + "="*85)
        print(f"ACTIVATION DISTRIBUTIONS (top-{top_n} by average rank, same order as table above)")
        print("="*85)
        for i, feat in enumerate(candidates_by_avg[:top_n], 1):
            L = feat['layer_idx']
            if L not in positive_activations or L not in negative_activations:
                print(f"  {i:3d}. Layer {L}, Feature {feat['feature_idx']} — activations unavailable for this layer")
                continue
            marker = " ◄ TARGET" if (L == target_layer and feat['feature_idx'] == target_feature) else ""
            print_activation_dist(i, feat, positive_activations[L], negative_activations[L],
                                  n_bins=args.dist_bins, marker=marker)
        # Also show target if it fell outside the top-N
        if target_feat is not None:
            in_topN = any(f['layer_idx'] == target_layer and f['feature_idx'] == target_feature
                          for f in candidates_by_avg[:top_n])
            if not in_topN and target_layer in positive_activations and target_layer in negative_activations:
                print()
                print_activation_dist(0, target_feat, positive_activations[target_layer],
                                      negative_activations[target_layer],
                                      n_bins=args.dist_bins, marker=" ◄ TARGET (outside top-N)")

    print(f"\n✓ Feature discovery complete!")
    discovery.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
