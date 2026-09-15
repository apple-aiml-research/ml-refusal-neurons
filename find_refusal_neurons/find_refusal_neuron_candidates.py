#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
find_refusal_neuron_candidates.py — Contrastive gradient × activation ranking

For each MLP feature and each selected token position, accumulate:
  - H_grad[t], N_grad[t]: mean signed gradient over harmful / harmless prompts
  - H_act[t],  N_act[t]:  mean activation value over harmful / harmless prompts

Score per (feature, token):
  G[t]     = H_grad[t] + N_grad[t]            (combined gradient signal)
  score[t] = G[t] * (N_act[t] - H_act[t])    (gradient × activation gap)

Best token for feature f: t* = argmax_t(score[t])
Feature score: score[t*]
Sorted descending.

Usage:
    python find_refusal_neurons/find_refusal_neuron_candidates.py \\
        --model Qwen/Qwen3-14B --loss log_odds \\
        --selected_tokens="-5,-6,-7,-8,-9" --top_k 20 \\
        --harmful_path data/splits/harmful_train.json \\
        --harmless_path data/splits/harmless_train.json \\
        --target_phrases "I'm sorry, but I can't help with that request.;I'm unable to assist" \\
        --show_features "17:2154" --prune_last 0.33 --magnitude_norm
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load_prompts(harmful_path=None, harmless_path=None):
    if harmful_path is None:
        harmful_path = os.path.join(DATA_DIR, "harmful_prompts.json")
    if harmless_path is None:
        harmless_path = os.path.join(DATA_DIR, "harmless_prompts.json")
    with open(harmful_path) as f:
        harmful = json.load(f)
    with open(harmless_path) as f:
        harmless = json.load(f)
    return ([item["text"] for item in harmful],
            [item["text"] for item in harmless])


def phrase_log_prob(logits, phrase_ids):
    n = len(phrase_ids)
    log_p = 0.0
    for i, tid in enumerate(phrase_ids):
        pos = -(n - i + 1)
        lp = torch.log_softmax(logits[pos].float(), dim=-1)
        log_p = log_p + lp[tid]
    return log_p


def accumulate(model, tok, prompts, target_phrases_ids, layers_set,
               selected_tokens, loss_type):
    """
    Returns (grad_sums, act_sums), each a dict {layer_idx: np.array [n_sel, n_features]}.
    grad_sums: sum of signed gradients over all prompts at each selected token.
    act_sums:  sum of activation values over all prompts at each selected token.
    """
    n_features = {}
    for name, m in model.named_modules():
        parts = name.split('.')
        if name.endswith('mlp') and len(parts) >= 3 and parts[2].isdigit():
            li = int(parts[2])
            if li in layers_set:
                n_features[li] = m.gate_proj.out_features

    embed_device = model.model.embed_tokens.weight.device
    n_sel = len(selected_tokens)

    # Keep sums on GPU as float32 tensors; convert to numpy only at the end
    grad_sums = {li: torch.zeros(n_sel, nf, dtype=torch.float32, device=embed_device)
                 for li, nf in n_features.items()}
    act_sums  = {li: torch.zeros(n_sel, nf, dtype=torch.float32, device=embed_device)
                 for li, nf in n_features.items()}

    captured = {li: [] for li in layers_set}
    handles = []

    for name, m in model.named_modules():
        parts = name.split('.')
        if name.endswith('mlp') and len(parts) >= 3 and parts[2].isdigit():
            li = int(parts[2])
            if li not in layers_set:
                continue

            def make_hook(layer_idx):
                def hook_fn(module, inp):
                    # inp[0] is the already-computed gated activation — no recomputation
                    gated = inp[0]
                    gated.retain_grad()
                    is_first = (len(captured[layer_idx]) == 0)
                    captured[layer_idx].append((gated, is_first))
                return hook_fn

            handles.append(m.down_proj.register_forward_pre_hook(make_hook(li)))

    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx % 20 == 0:
            print(f"    {prompt_idx + 1}/{len(prompts)}", flush=True)

        for li in captured:
            captured[li].clear()

        msgs = [{'role': 'user', 'content': prompt}]
        try:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

        input_ids = tok(text, return_tensors='pt',
                        add_special_tokens=False).input_ids.to(embed_device)
        seq_len = input_ids.shape[1]

        # Precompute valid (t_offset, abs_pos) pairs once per prompt
        valid_positions = []
        for t_offset, t in enumerate(selected_tokens):
            abs_pos = seq_len + t
            if 0 <= abs_pos < seq_len:
                valid_positions.append((t_offset, abs_pos))

        with torch.no_grad():
            input_embeds = model.model.embed_tokens(input_ids)
        input_embeds = input_embeds.detach().requires_grad_(True)

        all_lps = []
        for phrase_ids in target_phrases_ids:
            phrase_tensor = torch.tensor(phrase_ids, device=embed_device).unsqueeze(0)
            with torch.no_grad():
                phrase_embeds = model.model.embed_tokens(phrase_tensor)
            full_embeds = torch.cat([input_embeds, phrase_embeds.detach()], dim=1)
            full_attn = torch.ones(1, full_embeds.shape[1], device=full_embeds.device)
            out = model(inputs_embeds=full_embeds, attention_mask=full_attn)
            lp = phrase_log_prob(out.logits[0], phrase_ids)
            all_lps.append(lp)

        refusal_lp = torch.stack(all_lps).logsumexp(dim=0)
        if loss_type == "log_odds":
            comply_lp = torch.log1p(-refusal_lp.exp().clamp(max=1 - 1e-6))
            loss = -(refusal_lp - comply_lp)
        else:
            loss = -refusal_lp

        loss.backward()

        for li in layers_set:
            for gated, is_first in captured[li]:
                # Activations: only from first phrase pass
                if is_first:
                    act = gated.detach()[0]  # [seq_len, n_features]
                    for t_offset, abs_pos in valid_positions:
                        act_sums[li][t_offset].add_(act[abs_pos].float().to(embed_device))

                # Gradients: accumulate from all phrase passes
                if gated.grad is not None:
                    grad = gated.grad[0]  # [seq_len, n_features]
                    for t_offset, abs_pos in valid_positions:
                        grad_sums[li][t_offset].add_(grad[abs_pos].float().to(embed_device))

    for h in handles:
        h.remove()

    # Convert to numpy only once at the end
    return ({li: v.double().cpu().numpy() for li, v in grad_sums.items()},
            {li: v.double().cpu().numpy() for li, v in act_sums.items()})


def normalize(sums, n):
    return {li: arr / n for li, arr in sums.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Contrastive gradient x activation ranking")
    parser.add_argument("--model", required=True, help="HuggingFace model name/path, e.g. Qwen/Qwen3-14B")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to analyze, e.g. '10,11,12'. If omitted, analyzes all layers minus --prune_last.")
    parser.add_argument("--target_phrases", type=str, default="I'm unable to assist",
                        help="Semicolon-separated refusal phrases")
    parser.add_argument("--loss", type=str, default="log_odds",
                        choices=["log_odds", "neg_log_prob"],
                        help="Refusal loss to take the gradient of: 'log_odds' (default, matches the paper's Eq. 2) = -log(p/(1-p)); 'neg_log_prob' (not used in the paper) = -log(p). p is the total probability mass over --target_phrases.")
    parser.add_argument("--selected_tokens", type=str, required=True,
                        help="Comma-separated token positions (negative = from end) to score at. Should match the model's post-instruction suffix length: '-5,-6,-7,-8,-9' for Qwen3, '-2,-3,-4,-5' for Llama.")
    parser.add_argument("--top_k", type=int, default=50, help="Number of top-ranked features to print")
    parser.add_argument("--show_features", type=str, default=None,
                        help="Comma-separated layer:feature pairs, e.g. '17:2154,17:1000'")
    parser.add_argument("--harmful_path", type=str, default="data/splits/harmful_train.json",
                        help="Path to a flat JSON list of {'text': ...} harmful prompts")
    parser.add_argument("--harmless_path", type=str, default="data/splits/harmless_train.json",
                        help="Path to a flat JSON list of {'text': ...} harmless prompts")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Cap the number of harmful and harmless prompts loaded — useful for a quick smoke test before a full run")
    parser.add_argument("--output", type=str, default=None,
                        help="Save the full ranked feature list + metadata to this JSON path. If omitted, results are only printed to stdout.")
    parser.add_argument("--prune_last", type=float, default=None,
                        help="When --layers is not specified, exclude the last fraction of layers. "
                             "E.g. 0.33 removes the last 33%% of layers.")
    parser.add_argument("--sign_align", action="store_true",
                        help="Only keep features where sign(H_grad) == sign(N_grad) at best token")
    parser.add_argument("--magnitude_norm", action="store_true",
                        help="Filter out features where |N_act| > |H_act| at best token "
                             "(i.e. harmless activation magnitude exceeds harmful)")
    parser.add_argument("--grad_term", type=str, default="HN", choices=["HN", "H", "N"],
                        help="Which gradient terms to use in G: HN (default, sum both), H (harmful only), N (harmless only)")
    parser.add_argument("--token_choose_method", type=str, default="score",
                        choices=["score", "act_diff"],
                        help="How to pick best token per feature: "
                             "'score' (default) = argmax(G*(N_act-H_act)), "
                             "'act_diff' = argmax(N_act-H_act)")
    parser.add_argument("--score_mode", type=str, default="product",
                        choices=["product", "log_weighted"],
                        help="Scoring formula: 'product' (default) = G*(N_act-H_act); "
                             "'log_weighted' = alpha*log|G| + (1-alpha)*log|N_act-H_act|, "
                             "only where G*(N_act-H_act) > 0")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for log|G| term in log_weighted mode (default 0.5)")
    args = parser.parse_args()

    selected_tokens = [int(x.strip()) for x in args.selected_tokens.split(',')]

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    harmful, harmless = load_prompts(args.harmful_path, args.harmless_path)
    if args.max_prompts:
        harmful  = harmful[:args.max_prompts]
        harmless = harmless[:args.max_prompts]
    print(f"Prompts: {len(harmful)} harmful, {len(harmless)} harmless")

    target_phrases_ids = [
        tok.encode(p.strip(), add_special_tokens=False)
        for p in args.target_phrases.split(';')
    ]
    print(f"Target phrases: {[tok.decode(ids) for ids in target_phrases_ids]}")
    print(f"Selected tokens: {selected_tokens}  |  Loss: {args.loss}"
          + (f"  |  grad_term={args.grad_term}" if args.grad_term != "HN" else "")
          + ("  |  sign_align=ON" if args.sign_align else "")
          + ("  |  magnitude_norm=ON" if args.magnitude_norm else ""))

    # Show what the selected token positions correspond to on the first harmful prompt
    _msgs = [{'role': 'user', 'content': harmful[0]}]
    try:
        _text = tok.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        _text = tok.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=True)
    _ids = tok(_text, return_tensors='pt', add_special_tokens=False).input_ids[0]
    _seq_len = len(_ids)
    print(f"  (example from first harmful prompt, seq_len={_seq_len})")
    for t in selected_tokens:
        abs_pos = _seq_len + t
        if 0 <= abs_pos < _seq_len:
            token_str = repr(tok.decode([_ids[abs_pos].item()]))
            print(f"    pos {t:3d}  (abs {abs_pos:3d})  →  {token_str}")

    n_layers = model.config.num_hidden_layers
    if args.layers:
        layers_set = set(int(x) for x in args.layers.split(','))
    else:
        keep = n_layers - (int(n_layers * args.prune_last) if args.prune_last else 0)
        layers_set = set(range(keep))

    print("\nHarmful prompts...")
    h_grad, h_act = accumulate(
        model, tok, harmful, target_phrases_ids, layers_set, selected_tokens, args.loss)
    h_grad = normalize(h_grad, len(harmful))
    h_act  = normalize(h_act,  len(harmful))

    print("\nHarmless prompts...")
    n_grad, n_act = accumulate(
        model, tok, harmless, target_phrases_ids, layers_set, selected_tokens, args.loss)
    n_grad = normalize(n_grad, len(harmless))
    n_act  = normalize(n_act,  len(harmless))

    all_features = []
    for layer_idx in sorted(layers_set):
        H_grad = h_grad[layer_idx]   # [n_sel, n_features]
        N_grad = n_grad[layer_idx]
        H_act  = h_act[layer_idx]
        N_act  = n_act[layer_idx]

        G     = H_grad + N_grad          # [n_sel, n_features]
        if args.grad_term == "H":
            G = H_grad
        elif args.grad_term == "N":
            G = N_grad
        score = G * (N_act - H_act)      # [n_sel, n_features]

        if args.score_mode == "log_weighted":
            eps = 1e-30
            pos_mask = score > 0
            ranking_score = np.where(
                pos_mask,
                args.alpha * np.log(np.abs(G) + eps)
                + (1 - args.alpha) * np.log(np.abs(N_act - H_act) + eps),
                -np.inf,
            )
        else:
            ranking_score = score

        n_feat = H_grad.shape[1]

        # Vectorized: find best token index per feature [n_features]
        if args.token_choose_method == "act_diff":
            best = np.argmax(np.abs(N_act - H_act), axis=0)  # [n_features]
        else:
            best = np.argmax(ranking_score, axis=0)           # [n_features]

        fi = np.arange(n_feat)
        h_grad_b = H_grad[best, fi]
        n_grad_b = N_grad[best, fi]
        h_act_b  = H_act[best, fi]
        n_act_b  = N_act[best, fi]
        score_b  = score[best, fi]
        rscore_b = ranking_score[best, fi]
        tok_b    = np.array(selected_tokens)[best]

        if args.sign_align:
            keep = np.sign(h_grad_b) == np.sign(n_grad_b)
        else:
            keep = np.ones(n_feat, dtype=bool)

        if args.magnitude_norm:
            keep &= np.abs(h_act_b) >= np.abs(n_act_b)

        if args.score_mode == "log_weighted":
            g_b      = G[best, fi]
            dact_b   = n_act_b - h_act_b
            log_g_b  = np.log(np.abs(g_b)    + 1e-30)
            log_da_b = np.log(np.abs(dact_b) + 1e-30)

        for feat in np.where(keep)[0]:
            entry = {
                'layer_idx':     layer_idx,
                'feature_idx':   int(feat),
                'best_token':    int(tok_b[feat]),
                'h_grad':        float(h_grad_b[feat]),
                'n_grad':        float(n_grad_b[feat]),
                'h_act':         float(h_act_b[feat]),
                'n_act':         float(n_act_b[feat]),
                'score':         float(score_b[feat]),
                'ranking_score': float(rscore_b[feat]),
            }
            if args.score_mode == "log_weighted":
                entry['log_g']    = float(log_g_b[feat])
                entry['log_dact'] = float(log_da_b[feat])
            all_features.append(entry)

    all_features.sort(key=lambda x: x['ranking_score'], reverse=True)

    rank_label = "alpha*log|G|+(1-alpha)*log|N_act-H_act|" if args.score_mode == "log_weighted" else "G*(N_act-H_act)"
    log_mode = args.score_mode == "log_weighted"
    print(f"\n{'='*100}")
    print(f"TOP {args.top_k} features  (sorted by {rank_label}, descending)")
    if log_mode:
        print(f"  {'Layer':>5}  {'Feat':>7}  {'Tok':>4}  "
              f"{'log|G|':>11}  {'log|N-H|':>11}  {'H_act':>11}  {'N_act':>11}  {'Score':>11}  {'LogScore':>11}")
    else:
        print(f"  {'Layer':>5}  {'Feat':>7}  {'Tok':>4}  "
              f"{'H_grad':>11}  {'N_grad':>11}  {'H_act':>11}  {'N_act':>11}  {'Score':>11}")
    print(f"{'='*100}")
    for c in all_features[:args.top_k]:
        if log_mode:
            print(f"  {c['layer_idx']:5d}  {c['feature_idx']:7d}  {c['best_token']:4d}  "
                  f"{c['log_g']:11.3f}  {c['log_dact']:11.3f}  "
                  f"{c['h_act']:11.3e}  {c['n_act']:11.3e}  {c['score']:11.3e}  {c['ranking_score']:11.3f}")
        else:
            print(f"  {c['layer_idx']:5d}  {c['feature_idx']:7d}  {c['best_token']:4d}  "
                  f"{c['h_grad']:11.3e}  {c['n_grad']:11.3e}  "
                  f"{c['h_act']:11.3e}  {c['n_act']:11.3e}  {c['score']:11.3e}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({
                'metadata': {
                    'model': args.model, 'layers': args.layers,
                    'target_phrases': args.target_phrases, 'loss': args.loss,
                    'selected_tokens': args.selected_tokens,
                    'score_mode': args.score_mode, 'alpha': args.alpha,
                    'n_harmful': len(harmful), 'n_harmless': len(harmless),
                },
                'features': all_features,
            }, f, indent=2)
        print(f"\nSaved to {args.output}")

    if args.show_features:
        rank_lookup = {(c['layer_idx'], c['feature_idx']): i + 1
                       for i, c in enumerate(all_features)}
        feat_lookup = {(c['layer_idx'], c['feature_idx']): c
                       for c in all_features}
        print(f"\n{'='*100}")
        print("SPECIFIC FEATURES")
        if log_mode:
            print(f"  {'Layer':>5}  {'Feat':>7}  {'Rank':>6}  {'Tok':>4}  "
                  f"{'log|G|':>11}  {'log|N-H|':>11}  {'H_act':>11}  {'N_act':>11}  {'Score':>11}  {'LogScore':>11}")
        else:
            print(f"  {'Layer':>5}  {'Feat':>7}  {'Rank':>6}  {'Tok':>4}  "
                  f"{'H_grad':>11}  {'N_grad':>11}  {'H_act':>11}  {'N_act':>11}  {'Score':>11}")
        print(f"{'='*100}")
        for entry in args.show_features.split(','):
            layer_str, feat_str = entry.strip().split(':')
            key = (int(layer_str), int(feat_str))
            rank = rank_lookup.get(key, -1)
            c = feat_lookup.get(key)
            if c:
                if log_mode:
                    print(f"  {c['layer_idx']:5d}  {c['feature_idx']:7d}  {rank:6d}  "
                          f"{c['best_token']:4d}  {c['log_g']:11.3f}  {c['log_dact']:11.3f}  "
                          f"{c['h_act']:11.3e}  {c['n_act']:11.3e}  {c['score']:11.3e}  {c['ranking_score']:11.3f}")
                else:
                    print(f"  {c['layer_idx']:5d}  {c['feature_idx']:7d}  {rank:6d}  "
                          f"{c['best_token']:4d}  {c['h_grad']:11.3e}  {c['n_grad']:11.3e}  "
                          f"{c['h_act']:11.3e}  {c['n_act']:11.3e}  {c['score']:11.3e}")
            else:
                print(f"  {int(layer_str):5d}  {int(feat_str):7d}  not found in analyzed layers")


if __name__ == '__main__':
    main()
