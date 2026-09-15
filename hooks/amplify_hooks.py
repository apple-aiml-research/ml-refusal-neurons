#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Amplification Hook Utilities for Different Architectures

Supports amplifying/suppressing specific features during generation
for LLaMA and GPT-2 architectures.
"""

import torch


def parse_mult_intervals(spec):
    """
    Parse an interval multiplier spec string into a list of (start, stop, mult) tuples.

    Format: "start:stop:mult,start:stop:mult,..."
    Uses Python slice semantics for start/stop (None = beginning/end, negatives count from end).

    Examples:
        ":-4:30,-4::50"      -> [(None, -4, 30.0), (-4, None, 50.0)]
        ":50:-100,50::-50"   -> [(None, 50, -100.0), (50, None, -50.0)]
        "::-20"              -> [(None, None, -20.0)]   # all tokens

    Gaps (positions not covered by any interval) are left unmodified.

    Args:
        spec: Interval spec string, optionally wrapped in brackets.

    Returns:
        List of (start, stop, mult) tuples, or None if spec is None/empty.
    """
    if not spec:
        return None
    spec = spec.strip().strip('[]').strip('"\'')
    intervals = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(':')
        if len(pieces) != 3:
            raise ValueError(
                f"Invalid interval spec '{part}' — expected 'start:stop:mult' "
                f"(e.g. ':-4:30' or '-4::50')"
            )
        start = int(pieces[0]) if pieces[0] else None
        stop  = int(pieces[1]) if pieces[1] else None
        mult  = float(pieces[2])
        intervals.append((start, stop, mult))
    return intervals if intervals else None


def _apply_prefill_intervals(tensor, seq_len, intervals, feature_idx, additive):
    """Apply interval multipliers to a prefill tensor in-place."""
    for start, stop, mult in intervals:
        abs_start = (seq_len + start) if (start is not None and start < 0) else (start if start is not None else 0)
        abs_stop  = (seq_len + stop)  if (stop  is not None and stop  < 0) else (stop  if stop  is not None else seq_len)
        abs_start = max(0, min(abs_start, seq_len))
        abs_stop  = max(0, min(abs_stop,  seq_len))
        if abs_start >= abs_stop:
            continue
        if additive:
            tensor[:, abs_start:abs_stop, feature_idx] = tensor[:, abs_start:abs_stop, feature_idx] + mult
        else:
            tensor[:, abs_start:abs_stop, feature_idx] = mult


def _find_response_mult(intervals, token_count):
    """Return the multiplier for the current response token, or None if no interval covers it."""
    for start, stop, mult in intervals:
        abs_start = start if start is not None else 0
        if abs_start <= token_count and (stop is None or token_count < stop):
            return mult
    return None


def create_amplify_hook(
    architecture: str,
    layer_num: int,
    feature_idx: int,
    multiplier: float,
    amplify_last: int = None,
    response_mult: float = None,
    pre_down_proj: bool = False,
    post_down_proj: bool = False,
    response_amplify_tokens: int = None,
    response_tail_mult: float = None,
    p_intervals=None,
    r_intervals=None,
    additive: bool = True,
    negate: bool = False,
):
    """
    Create an amplification hook that modifies a specific feature during forward pass.

    When pre_down_proj=True (llama only):
        Registers a forward_pre_hook on the down_proj module.
        The input to down_proj is already act_fn(gate)*up — we modify the
        feature in-place and return it. No recomputation of any projections.

    When pre_down_proj=False (llama) or gpt2:
        Registers a forward_hook on the MLP module (original behaviour).

    Scalar API (original, fully preserved):
        multiplier + amplify_last                              — prefill
        response_mult + response_amplify_tokens + response_tail_mult  — response

    Interval API (new, overrides scalar when provided):
        p_intervals  — list of (start, stop, mult) for prefill tokens
                       negative indices count from end of prefill sequence
        r_intervals  — list of (start, stop, mult) for response tokens (positive indices)

    Gaps (token positions not covered by any interval) are left unmodified.

    Returns:
        (hook_fn, state_dict)
    """
    state = {'active': True, 'response_token_count': 0}

    if architecture == 'llama':

        if pre_down_proj:
            # ── Pre-hook on down_proj ────────────────────────────────────────
            # inp[0] is already act_fn(gate)*up  shape: (batch, seq_len, d_ff)
            # We modify the feature in-place and return (features,).
            # No projection is recomputed — zero extra overhead.
            def llama_down_proj_pre_hook(module, inp):
                if not state['active']:
                    return None

                features = inp[0]

                if negate:
                    features[:, :, feature_idx] = -features[:, :, feature_idx]
                    return (features,)

                if features.shape[1] != 1:
                    # ── Prefill ──────────────────────────────────────────────
                    state['response_token_count'] = 0
                    seq_len = features.shape[1]
                    if p_intervals is not None:
                        _apply_prefill_intervals(features, seq_len, p_intervals, feature_idx, additive=additive)
                    elif amplify_last is not None and amplify_last > 0:
                        if additive:
                            features[:, -amplify_last:, feature_idx] = features[:, -amplify_last:, feature_idx] + multiplier
                        else:
                            features[:, -amplify_last:, feature_idx] = multiplier
                    else:
                        if additive:
                            features[:, :, feature_idx] = features[:, :, feature_idx] + multiplier
                        else:
                            features[:, :, feature_idx] = multiplier
                else:
                    # ── Decoding ─────────────────────────────────────────────
                    cnt = state['response_token_count']
                    if r_intervals is not None:
                        m = _find_response_mult(r_intervals, cnt)
                        if m is not None:
                            if additive:
                                features[:, :, feature_idx] = features[:, :, feature_idx] + m
                            else:
                                features[:, :, feature_idx] = m
                    elif response_mult is not None:
                        if response_amplify_tokens is None or cnt < response_amplify_tokens:
                            if additive:
                                features[:, :, feature_idx] = features[:, :, feature_idx] + response_mult
                            else:
                                features[:, :, feature_idx] = response_mult
                        elif response_tail_mult is not None:
                            if additive:
                                features[:, :, feature_idx] = features[:, :, feature_idx] + response_tail_mult
                            else:
                                features[:, :, feature_idx] = response_tail_mult
                    state['response_token_count'] += 1

                return (features,)

            return llama_down_proj_pre_hook, state

        elif post_down_proj:
            # ── Post-hook on mlp output (post_down_proj=True) ────────────────
            # output is the MLP contribution to the residual stream: (batch, seq_len, d_model)
            # feature_idx is a d_model dimension (residual stream space).
            def llama_post_down_proj_hook(module, inp, output):
                if not state['active']:
                    return output

                if negate:
                    output[:, :, feature_idx] = -output[:, :, feature_idx]
                    return output

                if output.shape[1] != 1:
                    # ── Prefill ──────────────────────────────────────────────
                    state['response_token_count'] = 0
                    seq_len = output.shape[1]
                    if p_intervals is not None:
                        _apply_prefill_intervals(output, seq_len, p_intervals, feature_idx, additive=additive)
                    elif amplify_last is not None and amplify_last > 0:
                        if additive:
                            output[:, -amplify_last:, feature_idx] = output[:, -amplify_last:, feature_idx] + multiplier
                        else:
                            output[:, -amplify_last:, feature_idx] = multiplier
                    else:
                        if additive:
                            output[:, :, feature_idx] = output[:, :, feature_idx] + multiplier
                        else:
                            output[:, :, feature_idx] = multiplier
                else:
                    # ── Decoding ─────────────────────────────────────────────
                    cnt = state['response_token_count']
                    if r_intervals is not None:
                        m = _find_response_mult(r_intervals, cnt)
                        if m is not None:
                            if additive:
                                output[:, :, feature_idx] = output[:, :, feature_idx] + m
                            else:
                                output[:, :, feature_idx] = m
                    elif response_mult is not None:
                        if response_amplify_tokens is None or cnt < response_amplify_tokens:
                            if additive:
                                output[:, :, feature_idx] = output[:, :, feature_idx] + response_mult
                            else:
                                output[:, :, feature_idx] = response_mult
                        elif response_tail_mult is not None:
                            if additive:
                                output[:, :, feature_idx] = output[:, :, feature_idx] + response_tail_mult
                            else:
                                output[:, :, feature_idx] = response_tail_mult
                    state['response_token_count'] += 1

                return output

            return llama_post_down_proj_hook, state

        else:
            # ── Post-hook on mlp (pre_down_proj=False) ───────────────────────
            # Original replace-mode: recomputes gate/up projections, sets
            # up_features[feature_idx] = multiplier, gate[feature_idx] = 1.0
            def llama_amplify_hook(module, inp, output):
                if not state['active']:
                    return output

                inp = inp[0]
                gate_features = module.gate_proj(inp)
                up_features   = module.up_proj(inp)

                if negate:
                    up_features[:, :, feature_idx] = -up_features[:, :, feature_idx]
                    features = module.act_fn(gate_features) * up_features
                    return module.down_proj(features)

                if up_features.shape[1] != 1:
                    # ── Prefill ──────────────────────────────────────────────
                    state['response_token_count'] = 0
                    seq_len = up_features.shape[1]
                    if p_intervals is not None:
                        _apply_prefill_intervals(up_features, seq_len, p_intervals, feature_idx, additive=False)
                        for start, stop, _ in p_intervals:
                            abs_start = (seq_len + start) if (start is not None and start < 0) else (start if start is not None else 0)
                            abs_stop  = (seq_len + stop)  if (stop  is not None and stop  < 0) else (stop  if stop  is not None else seq_len)
                            abs_start = max(0, min(abs_start, seq_len))
                            abs_stop  = max(0, min(abs_stop,  seq_len))
                            if abs_start < abs_stop:
                                gate_features[:, abs_start:abs_stop, feature_idx] = 1.0
                    elif amplify_last is not None and amplify_last > 0:
                        up_features[:, -amplify_last:, feature_idx]   = multiplier
                        gate_features[:, -amplify_last:, feature_idx] = 1.0
                    else:
                        up_features[:, :, feature_idx]   = multiplier
                        gate_features[:, :, feature_idx] = 1.0
                else:
                    # ── Decoding ─────────────────────────────────────────────
                    cnt = state['response_token_count']
                    if r_intervals is not None:
                        m = _find_response_mult(r_intervals, cnt)
                        if m is not None:
                            up_features[:, :, feature_idx]   = m
                            gate_features[:, :, feature_idx] = 1.0
                    elif response_mult is not None:
                        if response_amplify_tokens is None or cnt < response_amplify_tokens:
                            up_features[:, :, feature_idx]   = response_mult
                            gate_features[:, :, feature_idx] = 1.0
                        elif response_tail_mult is not None:
                            up_features[:, :, feature_idx]   = response_tail_mult
                            gate_features[:, :, feature_idx] = 1.0
                    state['response_token_count'] += 1

                features = module.act_fn(gate_features) * up_features
                output = module.down_proj(features)
                return output

            return llama_amplify_hook, state

    elif architecture == 'gpt2':
        def gpt2_amplify_hook(module, inp, output):
            if not state['active']:
                return output

            inp = inp[0]
            hidden = module.c_fc(inp)

            if hasattr(module, 'act'):
                features = module.act(hidden)
            elif hasattr(module, 'activation_fn'):
                features = module.activation_fn(hidden)
            else:
                features = torch.nn.functional.gelu(hidden)

            features[:, :, feature_idx] = multiplier
            output = module.c_proj(features)
            return output

        return gpt2_amplify_hook, state

    else:
        raise ValueError(f"Unsupported architecture for amplification: {architecture}")


def register_amplify_hook(
    model,
    architecture: str,
    layer_num: int,
    feature_idx: int,
    multiplier: float,
    amplify_last: int = None,
    response_mult: float = None,
    pre_down_proj: bool = False,
    post_down_proj: bool = False,
    response_amplify_tokens: int = None,
    response_tail_mult: float = None,
    p_intervals=None,
    r_intervals=None,
    additive: bool = True,
    negate: bool = False,
):
    """
    Register amplification hook on a specific layer.

    When pre_down_proj=True (llama): registers forward_pre_hook on down_proj.
    When post_down_proj=True (llama): registers forward_hook on mlp (d_model output space).
    Otherwise: registers forward_hook on the mlp module (original behaviour).

    Returns:
        Tuple of (hook_handle, state_dict, layer_name)
    """
    hook_fn, state = create_amplify_hook(
        architecture, layer_num, feature_idx, multiplier,
        amplify_last, response_mult, pre_down_proj, post_down_proj,
        response_amplify_tokens, response_tail_mult,
        p_intervals, r_intervals, additive, negate,
    )

    for name, module in model.named_modules():
        if architecture == 'llama':
            if name.endswith("mlp") and hasattr(module, 'gate_proj'):
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        if int(parts[i + 1]) == layer_num:
                            if pre_down_proj:
                                handle = module.down_proj.register_forward_pre_hook(hook_fn)
                            else:
                                handle = module.register_forward_hook(hook_fn)
                            return handle, state, name

        elif architecture == 'gpt2':
            if ('mlp' in name or 'MLP' in name) and hasattr(module, 'c_fc'):
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'h' and i + 1 < len(parts):
                        try:
                            if int(parts[i + 1]) == layer_num:
                                handle = module.register_forward_hook(hook_fn)
                                return handle, state, name
                        except ValueError:
                            continue

    raise ValueError(f"Could not find layer {layer_num} in model")
