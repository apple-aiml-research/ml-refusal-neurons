#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Interactive neuron intervention REPL.

Loads the model once, then lets you chat with the hook active, change settings
on the fly, and switch to any ranked neuron from the pre-computed rankings.

Two intervention modes:
  constant  — pin h_i = multiplier at every token (default)
  anchor    — capture neuron activation v at reference token(s), then apply
              h_i = clamp(v * m + m2, best_mult) during generation.

Anchor parameters (when using --rank or /set rank):
  m and m2 are derived from data/rankings.json using:
    d    = h_act - n_act          (harmful minus harmless mean activation)
    m2   = -d                     (shifts harmful activation to ~0)
    m    = best_mult / d * scale  (scales to reach best_mult at 1x)
  where h_act, n_act, best_mult are stored per neuron in rankings.json.
  When specifying --layer/--feature manually, set --m and --m2 explicitly.

Usage:
    python chat.py --model Qwen/Qwen3-14B --rank 1
    python chat.py --model Qwen/Qwen3-14B --rank 1 --anchor
    python chat.py --model Qwen/Qwen3-14B --layer 17 --feature 2154 --multiplier 40
    python chat.py --model Qwen/Qwen3-14B --layer 17 --feature 2154 --anchor --m -11.87 --m2 6.74

Commands:
  <prompt>                Generate with hook active
  /baseline <prompt>      Generate without hook
  /set rank <n>           Load rank-n neuron from rankings (sets layer, feature, multiplier/anchor params)
  /set anchor_scale <n>   Switch anchor mode, scale = n (auto-computes m from d values)
  /set mode constant      Switch to constant intervention mode
  /set layer <n>          Change layer
  /set feature <n>        Change feature
  /set mult <f>           Change multiplier (constant mode)
  /set m <f>              Change anchor m parameter
  /set m2 <f>             Change anchor m2 parameter
  /set tokens <n>         Change max_tokens
  /set temp <f>           Change temperature
  /status                 Show current settings
  /help                   Show this help
  /quit                   Exit
"""

import argparse
import json
import os
import sys
import threading

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hooks.amplify_hooks import register_amplify_hook
from hooks.model_hooks import detect_architecture

RANKINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rankings.json")

MODEL_KEY_MAP = {
    "qwen3-1.7b": "qwen1.7b",
    "qwen3-4b":   "qwen4b",
    "qwen3-8b":   "qwen8b",
    "qwen3-14b":  "qwen14b",
    "qwen3-32b":  "qwen32b",
    "llama-3.1-8b":  "llama8b",
    "llama-3.1-70b": "llama70b",
    "meta-llama-3.1-8b-instruct":  "llama8b",
    "meta-llama-3.1-70b-instruct": "llama70b",
}

# Anchor token positions differ by chat-template length (CLAUDE.md gotcha):
# Qwen3's template has a longer post-instruction suffix than Llama-3's.
ANCHOR_TOKEN_POSITIONS_BY_KEY = {
    "qwen1.7b": [-5, -6, -7, -8, -9], "qwen4b": [-5, -6, -7, -8, -9],
    "qwen8b":   [-5, -6, -7, -8, -9], "qwen14b": [-5, -6, -7, -8, -9],
    "qwen32b":  [-5, -6, -7, -8, -9],
    "llama8b":  [-2, -3, -4, -5], "llama70b": [-2, -3, -4, -5],
}
ANCHOR_TOKEN_POSITIONS_DEFAULT = [-5, -6, -7, -8, -9]


def anchor_token_positions(model_key):
    return ANCHOR_TOKEN_POSITIONS_BY_KEY.get(model_key, ANCHOR_TOKEN_POSITIONS_DEFAULT)


def anchor_token_agg(best_mult):
    """Paper Eq. 6: aggregate with max when d>0, min when d<0. Since best_mult and d
    always have opposite sign (the paper picks m* opposite the harmful direction),
    this is equivalent to: max for negative best_mult, min for positive best_mult."""
    if best_mult is None:
        return "min"
    return "max" if best_mult < 0 else "min"


def detect_model_key(model_name):
    lower = model_name.lower().split("/")[-1]
    for pattern, key in MODEL_KEY_MAP.items():
        if pattern in lower:
            return key
    return None


def load_rankings(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    for key in data:
        data[key] = sorted(data[key], key=lambda e: e["hb191_asr"], reverse=True)
        for i, e in enumerate(data[key]):
            e["rank"] = i + 1
    return data


def anchor_params_from_entry(entry, scale=1):
    """Compute m, m2, best_mult from rankings entry using d = h_act - n_act."""
    h_act = entry["h_act"]
    n_act = entry["n_act"]
    best_mult = entry["best_mult"]
    d = h_act - n_act
    m2 = -d                       # = n_act - h_act
    m_base = best_mult / d if d != 0 else 0.0
    m = m_base * scale
    return m, m2, best_mult


# ---------------------------------------------------------------------------
# Anchor hook (self-contained, no hooks/ dependency)
# ---------------------------------------------------------------------------

def register_anchor_hook(model, layer_num, feature_idx):
    state = {"active": False, "anchor_value": 0.0, "m": 1.0, "m2": 0.0, "best_mult": None}

    def hook_fn(module, inp):
        if not state["active"]:
            return None
        features = inp[0]
        G = state["anchor_value"] * state["m"] + state["m2"]
        bm = state["best_mult"]
        if bm is not None:
            G = min(G, bm) if bm > 0 else max(G, bm)
        features[:, :, feature_idx] = G
        return (features,)

    for name, module in model.named_modules():
        if name.endswith("mlp") and hasattr(module, "gate_proj"):
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    if int(parts[i + 1]) == layer_num:
                        handle = module.down_proj.register_forward_pre_hook(hook_fn)
                        return handle, state, name
    raise ValueError(f"Could not find MLP at layer {layer_num}")


def capture_activation(model, layer_num, feature_idx, input_ids):
    captured = {}

    def hook_fn(module, inp):
        captured["act"] = inp[0].detach()

    for name, module in model.named_modules():
        if name.endswith("mlp") and hasattr(module, "gate_proj"):
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    if int(parts[i + 1]) == layer_num:
                        handle = module.down_proj.register_forward_pre_hook(hook_fn)
                        with torch.no_grad():
                            model(input_ids)
                        handle.remove()
                        return captured["act"][0, :, feature_idx]
    raise ValueError(f"Could not find MLP at layer {layer_num}")


def set_anchor_value(model, hook_state, layer, feature, input_ids, token_positions, token_agg):
    acts = capture_activation(model, layer, feature, input_ids)
    seq_len = acts.shape[0]
    vals = []
    for tp in token_positions:
        ap = seq_len + tp if tp < 0 else tp
        if 0 <= ap < seq_len:
            vals.append(acts[ap].item())
    if not vals:
        return 0.0
    anchor_val = min(vals) if token_agg == "min" else max(vals)
    hook_state["anchor_value"] = anchor_val
    return anchor_val


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class _StopOnSignal(StoppingCriteria):
    def __init__(self):
        self.stop = False
    def __call__(self, input_ids, scores, **kwargs):
        return self.stop


def build_text(tokenizer, prompt, enable_thinking=False):
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def _run_generate(model, tokenizer, inputs, max_tokens, temperature, streaming):
    if streaming:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        stop_signal = _StopOnSignal()
        thread = threading.Thread(
            target=lambda: model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
                stopping_criteria=StoppingCriteriaList([stop_signal]),
            ), daemon=True)
        thread.start()
        try:
            for token in streamer:
                print(token, end="", flush=True)
        except KeyboardInterrupt:
            stop_signal.stop = True
            print("\n[stopped]", flush=True)
        finally:
            print(flush=True)
            thread.join(timeout=10)
    else:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(outputs[0], skip_special_tokens=False))


def do_generate(model, tokenizer, prompt, max_tokens, temperature, streaming,
                const_state, anchor_state, active, enable_thinking,
                anchor_mode=False, layer=None, feature=None, token_positions=None):
    text = build_text(tokenizer, prompt, enable_thinking)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # deactivate both; then activate only the appropriate one
    if const_state is not None:
        const_state["active"] = False
    if anchor_state is not None:
        anchor_state["active"] = False

    if active:
        if anchor_mode:
            token_agg = anchor_token_agg(anchor_state.get("best_mult"))
            set_anchor_value(model, anchor_state, layer, feature, inputs["input_ids"],
                              token_positions, token_agg)
            anchor_state["active"] = True
        else:
            if const_state is not None:
                const_state["active"] = True

    _run_generate(model, tokenizer, inputs, max_tokens, temperature, streaming)

    if const_state is not None:
        const_state["active"] = False
    if anchor_state is not None:
        anchor_state["active"] = False


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def print_status(model_name, layer, feature, multiplier, max_tokens, temperature,
                 additive, enable_thinking, model_key, rankings,
                 anchor_mode, anchor_scale, anchor_m, anchor_m2, anchor_best_mult):
    rank_info = ""
    if model_key and model_key in rankings:
        match = next((e for e in rankings[model_key]
                      if e["layer"] == layer and e["feature"] == feature), None)
        if match:
            rank_info = f"  [rerank #{match['rank']} — HB191 ASR: {match['hb191_asr']:.1%}]"

    if anchor_mode:
        mode_str = (f"anchor  scale={anchor_scale}x  m={anchor_m:.4f}"
                    f"  m2={anchor_m2:.4f}  best_mult={anchor_best_mult}"
                    f"  token_pos={anchor_token_positions(model_key)}"
                    f"  agg={anchor_token_agg(anchor_best_mult)}")
    else:
        mode_str = f"constant  mult={multiplier}  additive={additive}"

    print(f"\n  model={model_name}  layer={layer}  feature={feature}"
          f"  tokens={max_tokens}  temp={temperature}  think={enable_thinking}"
          f"\n  mode={mode_str}" + rank_info + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive neuron intervention REPL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--feature", type=int, default=None)
    parser.add_argument("--multiplier", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--architecture", type=str, default="auto", choices=["auto", "llama", "gpt2"])
    parser.add_argument("--pre_down_proj", action="store_true", default=True)
    parser.add_argument("--additive", default=False, action="store_true")
    parser.add_argument("--no_additive", dest="additive", action="store_false")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--rankings", type=str, default=RANKINGS_PATH)
    parser.add_argument("--rank", type=int, default=None,
                        help="Load this ASR-rank at startup")
    parser.add_argument("--anchor", action="store_true",
                        help="Start in anchor mode (requires --rank or rankings entry)")
    parser.add_argument("--anchor_scale", type=float, default=2,
                        help="Anchor scale multiplier (default: 2; paper evaluates 1x/2x but any value can be set)")
    parser.add_argument("--m", type=float, default=None,
                        help="Anchor m parameter (scale factor applied to captured activation)")
    parser.add_argument("--m2", type=float, default=None,
                        help="Anchor m2 parameter (offset added after scaling)")
    args = parser.parse_args()

    if args.rank is None:
        if args.layer is None or args.feature is None:
            parser.error("either --rank, or --layer/--feature (plus --multiplier for constant "
                         "mode or --m/--m2 for anchor mode), are required")
        if args.anchor:
            if args.m is None or args.m2 is None:
                parser.error("--anchor without --rank requires --m and --m2")
        elif args.multiplier is None:
            parser.error("constant mode without --rank requires --multiplier")

    layer      = args.layer or 0
    feature    = args.feature or 0
    multiplier = args.multiplier or 0.0
    max_tokens = args.max_tokens
    temperature = args.temperature
    additive   = args.additive
    enable_thinking = args.think
    streaming  = args.streaming
    anchor_mode  = args.anchor
    anchor_scale = args.anchor_scale
    anchor_m, anchor_m2, anchor_best_mult = args.m or 0.0, args.m2 or 0.0, None

    rankings  = load_rankings(args.rankings)
    model_key = detect_model_key(args.model)
    if model_key:
        print(f"Rankings key: {model_key} ({len(rankings.get(model_key, []))} entries)")
    else:
        print("Warning: could not detect model key — /set rank will not work")

    def apply_rank(rank_n, scale=1, use_anchor=None):
        nonlocal layer, feature, multiplier, anchor_m, anchor_m2, anchor_best_mult, anchor_mode, anchor_scale
        if not model_key or model_key not in rankings:
            print("  No rankings available for this model.")
            return False
        entry = next((e for e in rankings[model_key] if e["rank"] == rank_n), None)
        if entry is None:
            print(f"  Rank {rank_n} not found. Available: 1–{len(rankings[model_key])}")
            return False
        layer   = entry["layer"]
        feature = entry["feature"]
        multiplier = entry["best_mult"]
        anchor_scale = scale
        anchor_m, anchor_m2, anchor_best_mult = anchor_params_from_entry(entry, scale)
        if use_anchor is not None:
            anchor_mode = use_anchor
        print(f"  Loaded rerank #{rank_n}: L{layer} F{feature}  HB191 ASR={entry['hb191_asr']:.1%}")
        print(f"    constant: mult={multiplier}")
        print(f"    anchor {scale}x: m={anchor_m:.4f}  m2={anchor_m2:.4f}  best_mult={anchor_best_mult}")
        return True

    if args.rank is not None:
        apply_rank(args.rank, anchor_scale, use_anchor=anchor_mode)

    print(f"Loading {args.model} ...")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    architecture = args.architecture
    if architecture == "auto":
        architecture = detect_architecture(model)
        print(f"Architecture: {architecture}")

    # We keep two hooks: constant (from hooks/) and anchor (self-contained)
    const_handle = None
    const_state  = None
    anchor_handle = None
    anchor_state  = None

    def register_constant(l, f, m):
        nonlocal const_handle, const_state
        if const_handle is not None:
            const_handle.remove()
        const_handle, const_state, layer_name = register_amplify_hook(
            model, architecture, l, f, m,
            pre_down_proj=True, additive=additive,
            response_mult=m)
        const_state["active"] = False
        print(f"  Constant hook: layer={l} feature={f} mult={m} ({layer_name})")

    def register_anchor(l, f):
        nonlocal anchor_handle, anchor_state
        if anchor_handle is not None:
            anchor_handle.remove()
        anchor_handle, anchor_state, layer_name = register_anchor_hook(model, l, f)
        anchor_state["m"]         = anchor_m
        anchor_state["m2"]        = anchor_m2
        anchor_state["best_mult"] = anchor_best_mult
        anchor_state["active"]    = False
        print(f"  Anchor hook: layer={l} feature={f}  m={anchor_m:.4f}  m2={anchor_m2:.4f}  best_mult={anchor_best_mult} ({layer_name})")

    register_constant(layer, feature, multiplier)
    register_anchor(layer, feature)

    print_status(args.model, layer, feature, multiplier, max_tokens, temperature,
                 additive, enable_thinking, model_key, rankings,
                 anchor_mode, anchor_scale, anchor_m, anchor_m2, anchor_best_mult)
    print("Type a prompt to chat, or /help for commands.\n")

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not line:
            continue

        if line in ("/quit", "/exit"):
            print("Exiting.")
            break

        elif line == "/help":
            print(__doc__)

        elif line == "/status":
            print_status(args.model, layer, feature, multiplier, max_tokens, temperature,
                         additive, enable_thinking, model_key, rankings,
                         anchor_mode, anchor_scale, anchor_m, anchor_m2, anchor_best_mult)

        elif line.startswith("/baseline "):
            prompt = line[len("/baseline "):].strip()
            print("\n--- BASELINE ---")
            do_generate(model, tokenizer, prompt, max_tokens, temperature, streaming,
                        const_state, anchor_state, active=False, enable_thinking=enable_thinking)
            print()

        elif line.startswith("/set "):
            parts = line.split(None, 2)
            if len(parts) != 3:
                print("  Usage: /set <key> <value>")
                continue
            key, val = parts[1], parts[2]
            try:
                if key == "rank":
                    if apply_rank(int(val), anchor_scale):
                        register_constant(layer, feature, multiplier)
                        anchor_state["m"]         = anchor_m
                        anchor_state["m2"]        = anchor_m2
                        anchor_state["best_mult"] = anchor_best_mult
                        register_anchor(layer, feature)

                elif key == "anchor_scale":
                    scale = float(val)
                    anchor_scale = scale
                    # recompute from rankings if possible
                    entry = None
                    if model_key and model_key in rankings:
                        entry = next((e for e in rankings[model_key]
                                      if e["layer"] == layer and e["feature"] == feature), None)
                    if entry:
                        anchor_m, anchor_m2, anchor_best_mult = anchor_params_from_entry(entry, scale)
                    else:
                        print("  No rankings entry for current layer/feature — cannot auto-compute m")
                        continue
                    anchor_mode = True
                    anchor_state["m"]         = anchor_m
                    anchor_state["m2"]        = anchor_m2
                    anchor_state["best_mult"] = anchor_best_mult
                    print(f"  Anchor {scale}x: m={anchor_m:.4f}  m2={anchor_m2:.4f}  best_mult={anchor_best_mult}")

                elif key == "mode":
                    if val == "anchor":
                        anchor_mode = True
                        print(f"  Mode: anchor")
                    elif val == "constant":
                        anchor_mode = False
                        print(f"  Mode: constant  mult={multiplier}")
                    else:
                        print(f"  Unknown mode: {val}. Use 'anchor' or 'constant'.")

                elif key == "layer":
                    layer = int(val)
                    register_constant(layer, feature, multiplier)
                    register_anchor(layer, feature)
                elif key == "feature":
                    feature = int(val)
                    register_constant(layer, feature, multiplier)
                    register_anchor(layer, feature)
                elif key == "mult":
                    multiplier = float(val)
                    register_constant(layer, feature, multiplier)
                elif key == "m":
                    anchor_m = float(val)
                    anchor_state["m"] = anchor_m
                    print(f"  anchor m={anchor_m}")
                elif key == "m2":
                    anchor_m2 = float(val)
                    anchor_state["m2"] = anchor_m2
                    print(f"  anchor m2={anchor_m2}")
                elif key == "tokens":
                    max_tokens = int(val); print(f"  max_tokens={max_tokens}")
                elif key == "temp":
                    temperature = float(val); print(f"  temperature={temperature}")
                elif key == "think":
                    enable_thinking = bool(int(val)); print(f"  think={enable_thinking}")
                else:
                    print(f"  Unknown setting: {key}. Type /help.")
            except ValueError:
                print(f"  Invalid value: {val}")

        elif line.startswith("/"):
            print("  Unknown command. Type /help.")

        else:
            mode_label = (f"anchor scale={anchor_scale}x" if anchor_mode
                          else f"constant mult={multiplier}")
            print(f"\n--- layer={layer} feat={feature} {mode_label} ---")
            do_generate(model, tokenizer, line, max_tokens, temperature, streaming,
                        const_state, anchor_state, active=True, enable_thinking=enable_thinking,
                        anchor_mode=anchor_mode,
                        layer=layer if anchor_mode else None,
                        feature=feature if anchor_mode else None,
                        token_positions=anchor_token_positions(model_key) if anchor_mode else None)
            print()

    for h in [const_handle, anchor_handle]:
        if h is not None:
            h.remove()


if __name__ == "__main__":
    main()
