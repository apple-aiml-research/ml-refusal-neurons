# AGENTS.md

Agent reference. Read this before editing any file.

## What this repo is

Reproduction code for **"A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language Models"** (https://arxiv.org/abs/2605.08513).

Core claim: suppressing a single MLP neuron (the *refusal neuron*) is sufficient for high attack success rate (ASR) on 7 safety-aligned LLMs across two families (Qwen3 1.7B–32B, Llama-3.1 8B / 70B).

---

## User-facing docs (read these first)

For pipeline commands, model → neuron tables, chat REPL, and CLI flags, defer to the READMEs — they are authoritative:

- `README.md` — Quick Start, finding refusal neurons, constant/anchor intervention generation, chat.py REPL, judging responses
- `find_concept_neurons/README.md` — concept-probing pipeline (generate → find)
- `max_activations/README.md` — DuckDB activation database + browser UI
- `detection/README.md` — detection benchmarks (XSTest / WildGuardMix / ToxicChat / OpenAI Moderation)

---

## Intervention mechanics (not in any README)

### Constant mode (`hooks/amplify_hooks.py`)

`forward_pre_hook` on `down_proj` (SwiGLU MLP). At every token:
```
h_i = multiplier
```
Both prefill and response tokens are modified — via `response_mult=multiplier`.

Required parameters:
- `pre_down_proj=True` — hook is on `down_proj` input (post gate×up), not MLP output
- `additive=False` — replace, not add
- `response_mult=multiplier` — **required** or the hook silently stops firing during response generation

### Anchor mode (`chat.py`, self-contained)

1. Capture pass (no generation): run model on prompt, record `v = min(activations at token positions [-5,-6,-7,-8,-9])` at the target neuron.
2. Generation: apply `h_i = clamp(v * m + m2, best_mult)` at every token.

Anchor params derived from `data/rankings.json`:
```
d    = h_act - n_act
m2   = -d                          # shifts harmless baseline to ~0
m    = (best_mult / d) * scale     # scale ∈ {1, 2} in the paper (k)
```
`best_mult` is a ceiling (positive) / floor (negative) in the clamp.

---

## Gotchas — DO NOT change without understanding

- **`response_mult=multiplier`** in `register_amplify_hook` — removing it silently disables the hook during response generation (prefill still runs).
- **`active_state` pattern in `chat.py`** — both constant and anchor hooks are always registered; only one is activated per generation via `state["active"]`. Do not deregister mid-session.
- **`rankings.json` `rank` field** — overwritten at load time by `chat.py` after sorting on `hb191_asr`. Do not rely on the stored value.
- **Llama vs Qwen3 token positions** — Llama uses `-2,-3,-4,-5`; Qwen3 uses `-5,-6,-7,-8,-9`. Difference is intentional (Qwen3 has a longer chat-template suffix, since it appends an empty `<think>\n\n</think>\n\n` block even with `enable_thinking=False`).
- **`capture_stage` support diverges** — `hooks/model_hooks.py` supports `{gated, up_proj, gate_proj, down_proj}`. `find_concept_neurons/discovery/individual_cv.py` additionally supports `residual` (implemented locally). Do not assume the sets match.
- **`find_concept_neurons.py --token_aggregation`** — CLI restricts to `max`/`min`/`mean` even though `individual_cv.py` supports more (`select`, `selected_tokens`, `max_over_last`, `mean_over_last`, `minmax_tokens`). Widen the CLI choices if you need them.
- **`generation/`** now lives at `find_concept_neurons/generation/` (moved from `clean/generation/`). Old paths may still appear in commit history.
- **Architecture detection** — `"llama"` covers all Qwen3 and Llama-3 models (both use SwiGLU + `gate_proj`). `"gpt2"` branch exists but is unused for any paper model.
- **Manual anchor mode has no `best_mult` clamp** — `chat.py --layer/--feature --anchor --m --m2` (without `--rank`) never sets `best_mult`, so the hook's clamp is skipped entirely (`if bm is not None`). Only `--rank` (which pulls `best_mult` from `rankings.json` alongside `m`/`m2`) applies the ceiling/floor. A manual anchor command with the "right" `m`/`m2` for a ranked neuron is not fully equivalent to `--rank` — it can produce larger-magnitude interventions than the paper's clamped version.
