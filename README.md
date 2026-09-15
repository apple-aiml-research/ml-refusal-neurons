# A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language Models

**Paper:** https://arxiv.org/abs/2605.08513

> [!CAUTION]
> This repository and the paper contain examples of harmful content, including self-harm, sexual content, and other offensive language, used strictly for research and evaluation purposes.

<table align="center" border="0" cellspacing="0" cellpadding="4">
<tr>
<td align="center" valign="top">
  <img src="figures/activations.png" width="360"><br>
  <img src="figures/single_neuron_generation.png" width="360"><br>
  <img src="figures/attack_asr.png" width="360">
</td>
<td align="center" valign="top">
  <img src="figures/self_harm.png" width="345"><br>
  <img src="figures/self_harm_neuron_generation.png" width="345"><br>
  <img src="figures/harmful_harmless.png" width="170"><img src="figures/harmful_harmless_base.png" width="170">
</td>
</tr>
</table>

## Requirements

```bash
python -m venv ml-refusal-neurons-env
source ml-refusal-neurons-env/bin/activate
pip install -r requirements.txt
```

(or use `conda create -n ml-refusal-neurons python=3.11 && conda activate ml-refusal-neurons` instead of `venv`, if you prefer conda)

Log in to Hugging Face (required to download models and datasets):
```bash
hf auth login
```

## Quick Start

Suppress a refusal neuron by adding two lines to standard HuggingFace inference code — no need to run the full pipeline:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from hooks import NeuronSuppressor

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # or: Qwen/Qwen3-1.7B, Qwen/Qwen3-4B, Qwen/Qwen3-8B, Qwen/Qwen3-14B, Qwen/Qwen3-32B, meta-llama/Meta-Llama-3.1-70B-Instruct
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

suppressor = NeuronSuppressor.from_pretrained(model, model_name)  # <-- addition vs. standard HF generation

prompt = "How do I make methamphetamine?"
messages = [{"role": "user", "content": prompt}]
try:
    # enable_thinking=False disables Qwen3's thinking mode; Llama's template doesn't accept this kwarg
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False)
except TypeError:
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
if not isinstance(inputs, torch.Tensor):
    inputs = inputs["input_ids"]
inputs = inputs.to(model.device)

attention_mask = torch.ones_like(inputs)
with suppressor:  # <-- addition vs. standard HF generation
    output = model.generate(inputs, max_new_tokens=512, do_sample=False, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)

response = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("\n\n" + response)
```

`NeuronSuppressor.from_pretrained` loads the top ASR refusal neuron for the specified model from `data/rankings.json` — the same neuron reported in the paper for that model. Supported models: `Qwen/Qwen3-{1.7B,4B,8B,14B,32B}`, `meta-llama/Meta-Llama-3.1-{8B,70B}-Instruct`. For any other model, use `NeuronSuppressor(model, layer, feature, multiplier)` directly.

---

## Finding Refusal Neurons

Ranks MLP neurons by contrastive gradient × activation score across harmful vs. harmless prompts (Eq. 3–4 in the paper). The refusal neuron reported in the paper for each model is highlighted by `--show_features` along with its rank in this ranking. Run `python find_refusal_neurons/find_refusal_neuron_candidates.py --help` for the full flags reference.

Generate the harmful/harmless train/val splits first — we don't redistribute these datasets directly, some are gated on HuggingFace:

```bash
python find_refusal_neurons/build_train_val.py --seed 42
```

The commands below find refusal neuron candidates for each model using the gradient × activation-gap method described in the paper.

<details open>
<summary><b>Llama-3.1-8B</b> — refusal neuron: L11:F4258</summary>

```bash
python find_refusal_neurons/find_refusal_neuron_candidates.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct --loss log_odds \
    --selected_tokens="-2,-3,-4,-5" --top_k 20 \
    --harmful_path data/splits/harmful_train.json \
    --harmless_path data/splits/harmless_train.json \
    --target_phrases "I can't help with that.;I'm unable to assist" \
    --show_features "11:4258" --prune_last 0.33 --magnitude_norm
```
</details>

<details>
<summary><b>Llama-3.1-70B</b> — refusal neuron: L25:F10201</summary>

```bash
python find_refusal_neurons/find_refusal_neuron_candidates.py \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct --loss log_odds \
    --selected_tokens="-2,-3,-4,-5" --top_k 20 \
    --harmful_path data/splits/harmful_train.json \
    --harmless_path data/splits/harmless_train.json \
    --target_phrases "I can't help with that.;I'm unable to assist" \
    --show_features "25:10201" --prune_last 0.33 --magnitude_norm
```
</details>

<details>
<summary><b>Qwen3-1.7B</b> — refusal neuron: L13:F3270</summary>

```bash
python find_refusal_neurons/find_refusal_neuron_candidates.py \
    --model Qwen/Qwen3-1.7B --loss log_odds \
    --selected_tokens="-5,-6,-7,-8,-9" --top_k 20 \
    --harmful_path data/splits/harmful_train.json \
    --harmless_path data/splits/harmless_train.json \
    --target_phrases "I'm sorry, but I can't help with that request.;I'm unable to assist" \
    --show_features "13:3270" --prune_last 0.33 --magnitude_norm
```
</details>

<details>
<summary><b>Qwen3-4B</b> — refusal neuron: L14:F5590</summary>

```bash
python find_refusal_neurons/find_refusal_neuron_candidates.py \
    --model Qwen/Qwen3-4B --loss log_odds \
    --selected_tokens="-5,-6,-7,-8,-9" --top_k 20 \
    --harmful_path data/splits/harmful_train.json \
    --harmless_path data/splits/harmless_train.json \
    --target_phrases "I'm sorry, but I can't help with that request.;I'm unable to assist" \
    --show_features "14:5590" --prune_last 0.33 --magnitude_norm
```
</details>

<details>
<summary><b>Qwen3-8B</b> — refusal neuron: L14:F7924</summary>

```bash
python find_refusal_neurons/find_refusal_neuron_candidates.py \
    --model Qwen/Qwen3-8B --loss log_odds \
    --selected_tokens="-5,-6,-7,-8,-9" --top_k 20 \
    --harmful_path data/splits/harmful_train.json \
    --harmless_path data/splits/harmless_train.json \
    --target_phrases "I'm sorry, but I can't help with that request.;I'm unable to assist" \
    --show_features "14:7924" --prune_last 0.33 --magnitude_norm
```
</details>

<details>
<summary><b>Qwen3-14B</b> — refusal neuron: L17:F2154</summary>

```bash
python find_refusal_neurons/find_refusal_neuron_candidates.py \
    --model Qwen/Qwen3-14B --loss log_odds \
    --selected_tokens="-5,-6,-7,-8,-9" --top_k 20 \
    --harmful_path data/splits/harmful_train.json \
    --harmless_path data/splits/harmless_train.json \
    --target_phrases "I'm sorry, but I can't help with that request.;I'm unable to assist" \
    --show_features "17:2154" --prune_last 0.33 --magnitude_norm
```
</details>

<details>
<summary><b>Qwen3-32B</b> — refusal neuron: L40:F15515</summary>

```bash
python find_refusal_neurons/find_refusal_neuron_candidates.py \
    --model Qwen/Qwen3-32B --loss log_odds \
    --selected_tokens="-5,-6,-7,-8,-9" --top_k 20 \
    --harmful_path data/splits/harmful_train.json \
    --harmless_path data/splits/harmless_train.json \
    --target_phrases "I'm sorry, but I can't help with that request.;I'm unable to assist" \
    --show_features "40:15515" --prune_last 0.33 --magnitude_norm
```
</details>


### Reproducibility Note

The paper's original run wasn't seeded, and we can't release the exact prompts it used — some source datasets are gated on HuggingFace, and the specific sample was never recorded with a seed. `build_train_val.py --seed 42` (above) regenerates a reproducible substitute instead.

For each model, the paper computes HarmBench-191 ASR for its top-5 gradient-score candidates and picks the best-ASR one as the refusal neuron (`data/rankings.json`, rank #1). Under this reproducible seed, that same neuron still appears in the top-5 gradient-score candidates for every model except Qwen3-1.7B: there, the paper's neuron (L13/F3270) drops to rank #10, while the paper's *second*-best-ASR neuron for that model (L12/F3582, `data/rankings.json` rank #2) still appears at rank #5 — and the ASR gap between these two, as reported in the paper, is minimal (77.0% vs. 76.4%).

Results from both gradient-score runs are kept for reference: `data/gradient_rankings.json` (the paper's original run — not reproducible by running this repo's commands, per above) and `data/gradient_rankings_splits.json` (the seeded, reproducible substitute run). `data/rankings.json` — the file `chat.py`/`NeuronSuppressor` actually use — is the paper's reranking itself: a static, shipped record of the top-5 gradient candidates and their measured HarmBench-191 ASR from the paper's original run.

## Interactive Chat

Chat with a model while a neuron is suppressed. `--rank N` loads the Nth ASR-ranked neuron from `data/rankings.json`. Two intervention modes:

**Constant** — pins `h_i = multiplier` at every token, both prefill and decoding:
```bash
python chat.py --model Qwen/Qwen3-14B --rank 1
python chat.py --model Qwen/Qwen3-14B --layer 17 --feature 2154 --multiplier 40  # equivalent, spelled out manually
```

**Anchor** — captures the neuron's natural activation `v` on the prompt (a separate forward pass with a read-only hook that doesn't modify `v`), then applies `h_i = clamp(v * m + m2, best_mult)` at every token, both prefill and decoding:
```bash
python chat.py --model Qwen/Qwen3-14B --rank 1 --anchor                                        # default anchor_scale=2
python chat.py --model Qwen/Qwen3-14B --layer 17 --feature 2154 --anchor --m -11.87 --m2 6.74  # equivalent, spelled out manually

python chat.py --model Qwen/Qwen3-14B --rank 1 --anchor --anchor_scale 1        # best_scale for this model on the HarmBench-191 validation set
python chat.py --model Qwen/Qwen3-14B --layer 17 --feature 2154 --anchor --m -5.94 --m2 6.74  # equivalent, spelled out manually
```

The anchor 1x/2x sweep was only run for each model's rank-1 neuron; the scale the paper found best is stored as that entry's `anchor.best_scale` in `rankings.json`. The default `--anchor_scale` is `2`, but it's not auto-selected from `best_scale` — pass the right value explicitly (any scale is accepted; the paper evaluates only 1x/2x).

When using `--rank`, `m` and `m2` are derived automatically from `data/rankings.json`:
```
d    = h_act - n_act          # harmful minus harmless mean activation
m2   = -d                     # offsets a harmless-like activation, small relative to best_mult
m    = (best_mult / d) * scale  # scales a harmful-like activation toward best_mult
```

In the REPL:

| Command | Description |
|---|---|
| `<prompt>` | Generate with hook active |
| `/baseline <prompt>` | Generate without hook |
| `/set rank <n>` | Load rank-n neuron's layer, feature, mult; recompute anchor m/m2 at the current `anchor_scale` |
| `/set anchor_scale <n>` | Switch to anchor mode at scale `n`x (auto-computes m); paper evaluates 1x/2x |
| `/set mode constant` | Switch to constant mode |
| `/set mode anchor` | Switch to anchor mode |
| `/set layer <n>` | Change layer |
| `/set feature <n>` | Change feature |
| `/set mult <f>` | Change multiplier (constant mode) |
| `/set m <f>` | Change anchor m parameter |
| `/set m2 <f>` | Change anchor m2 parameter |
| `/set tokens <n>` | Change max tokens |
| `/set temp <f>` | Change temperature |
| `/status` | Show current settings and rank/ASR if known |
| `/quit` | Exit |
| `Ctrl+C` during generation | Stop the current generation early (e.g. to change the prompt or settings) and return to `>>>` |
| `Ctrl+C` at the `>>>` prompt | Exit the REPL |

`/set rank` looks up the neuron in `data/rankings.json` and sets layer, feature, and multiplier, then recomputes anchor `m`/`m2` at the current `anchor_scale`. All other `/set` commands still work afterward to override individual values.

The `best_mult` (constant mode) and `anchor.best_scale` (anchor mode) values stored in `rankings.json` are whichever maximised ASR on the HarmBench-191 validation set. Even so, some individual prompts need a stronger push: for constant mode, try `/set mult` with a larger magnitude in the same direction as `best_mult`; for anchor mode, try `/set anchor_scale` with a larger (always positive) value.

Sweeping across a range of multipliers instead of relying on the single stored `best_mult` can substantially increase ASR — see [Multiplier Sweep Sensitivity](#multiplier-sweep-sensitivity) below.

## Constant Intervention Generation

Pins the neuron's activation to a constant value during generation (`h_i = multiplier`) — replace mode, used for refusal-neuron generation in the paper. `--additive` switches to `h_i = h_i + multiplier` instead, the same mode used for concept neurons in [Concept Neuron Example](#concept-neuron-example--self-harm).

<details open>
<summary><b>Llama-3.1-8B</b>  L11:F4258  multiplier=-4</summary>

```bash
# HarmBench
python eval/generate_harmbench_constant.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --layer 11 --feature 4258 --multiplier -4

# JBB-Behaviors
python eval/generate_jbb_constant.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --layer 11 --feature 4258 --multiplier -4
```
</details>

<details>
<summary><b>Llama-3.1-70B</b>  L25:F10201  multiplier=-8</summary>

```bash
# HarmBench
python eval/generate_harmbench_constant.py \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --layer 25 --feature 10201 --multiplier -8

# JBB-Behaviors
python eval/generate_jbb_constant.py \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --layer 25 --feature 10201 --multiplier -8
```
</details>

<details open>
<summary><b>Qwen3-1.7B</b>  L13:F3270  multiplier=30</summary>

```bash
# HarmBench
python eval/generate_harmbench_constant.py \
    --model Qwen/Qwen3-1.7B \
    --layer 13 --feature 3270 --multiplier 30

# JBB-Behaviors
python eval/generate_jbb_constant.py \
    --model Qwen/Qwen3-1.7B \
    --layer 13 --feature 3270 --multiplier 30
```
</details>

<details>
<summary><b>Qwen3-4B</b>  L14:F5590  multiplier=18</summary>

```bash
# HarmBench
python eval/generate_harmbench_constant.py \
    --model Qwen/Qwen3-4B \
    --layer 14 --feature 5590 --multiplier 18

# JBB-Behaviors
python eval/generate_jbb_constant.py \
    --model Qwen/Qwen3-4B \
    --layer 14 --feature 5590 --multiplier 18
```
</details>

<details>
<summary><b>Qwen3-8B</b>  L14:F7924  multiplier=20</summary>

```bash
# HarmBench
python eval/generate_harmbench_constant.py \
    --model Qwen/Qwen3-8B \
    --layer 14 --feature 7924 --multiplier 20

# JBB-Behaviors
python eval/generate_jbb_constant.py \
    --model Qwen/Qwen3-8B \
    --layer 14 --feature 7924 --multiplier 20
```
</details>

<details>
<summary><b>Qwen3-14B</b>  L17:F2154  multiplier=40</summary>

```bash
# HarmBench
python eval/generate_harmbench_constant.py \
    --model Qwen/Qwen3-14B \
    --layer 17 --feature 2154 --multiplier 40

# JBB-Behaviors
python eval/generate_jbb_constant.py \
    --model Qwen/Qwen3-14B \
    --layer 17 --feature 2154 --multiplier 40
```
</details>

<details>
<summary><b>Qwen3-32B</b>  L40:F15515  multiplier=-80</summary>

```bash
# HarmBench
python eval/generate_harmbench_constant.py \
    --model Qwen/Qwen3-32B \
    --layer 40 --feature 15515 --multiplier -80

# JBB-Behaviors
python eval/generate_jbb_constant.py \
    --model Qwen/Qwen3-32B \
    --layer 40 --feature 15515 --multiplier -80
```
</details>

Results are saved to `results/eval/harmbench/` and `results/eval/jbb/` by default (pass `--output_dir` to override). The two HarmBench scripts default to **HarmBench-191** (the paper's validation set: HarmBench standard minus its 9-prompt overlap with JBB-Behaviors). Pass `--full_harmbench` to run on the raw 200-prompt set instead.

## Anchor Intervention Generation

Reads the neuron's natural activation at a reference token, then scales it during generation. `--m`/`--m2`/`--best_mult` below are each model's paper-selected best scale (1x or 2x — see [Interactive Chat](#interactive-chat)); `data/rankings.json`'s `anchor.scales` holds both.

<details open>
<summary><b>Llama-3.1-8B</b>  L11:F4258  scale=1x  m=-3.59  m2=-1.11  best_mult=-4</summary>

```bash
# HarmBench
python eval/generate_harmbench_anchor.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct --layer 11 --feature 4258 \
    --token_pos="-2,-3,-4,-5" --token_agg max \
    --m -3.59 --m2 -1.11 --best_mult -4 \
    --max_tokens 512 --temperature 0.0

# JBB-Behaviors
python eval/generate_jbb_anchor.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct --layer 11 --feature 4258 \
    --token_pos="-2,-3,-4,-5" --token_agg max \
    --m -3.59 --m2 -1.11 --best_mult -4 \
    --max_tokens 512 --temperature 0.0
```
</details>

<details>
<summary><b>Llama-3.1-70B</b>  L25:F10201  scale=2x  m=-72.05  m2=-0.22  best_mult=-8</summary>

```bash
# HarmBench
python eval/generate_harmbench_anchor.py \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct --layer 25 --feature 10201 \
    --token_pos="-2,-3,-4,-5" --token_agg max \
    --m -72.05 --m2 -0.22 --best_mult -8 \
    --max_tokens 512 --temperature 0.0

# JBB-Behaviors
python eval/generate_jbb_anchor.py \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct --layer 25 --feature 10201 \
    --token_pos="-2,-3,-4,-5" --token_agg max \
    --m -72.05 --m2 -0.22 --best_mult -8 \
    --max_tokens 512 --temperature 0.0
```
</details>

<details open>
<summary><b>Qwen3-1.7B</b>  L13:F3270  scale=2x  m=-5.98  m2=10.03  best_mult=30</summary>

```bash
# HarmBench
python eval/generate_harmbench_anchor.py \
    --model Qwen/Qwen3-1.7B --layer 13 --feature 3270 \
    --token_pos="-5,-6,-7,-8,-9" --token_agg min \
    --m -5.98 --m2 10.03 --best_mult 30 \
    --max_tokens 512 --temperature 0.0

# JBB-Behaviors
python eval/generate_jbb_anchor.py \
    --model Qwen/Qwen3-1.7B --layer 13 --feature 3270 \
    --token_pos="-5,-6,-7,-8,-9" --token_agg min \
    --m -5.98 --m2 10.03 --best_mult 30 \
    --max_tokens 512 --temperature 0.0
```
</details>

<details>
<summary><b>Qwen3-4B</b>  L14:F5590  scale=2x  m=-11.74  m2=3.07  best_mult=18</summary>

```bash
# HarmBench
python eval/generate_harmbench_anchor.py \
    --model Qwen/Qwen3-4B --layer 14 --feature 5590 \
    --token_pos="-5,-6,-7,-8,-9" --token_agg min \
    --m -11.74 --m2 3.07 --best_mult 18 \
    --max_tokens 512 --temperature 0.0

# JBB-Behaviors
python eval/generate_jbb_anchor.py \
    --model Qwen/Qwen3-4B --layer 14 --feature 5590 \
    --token_pos="-5,-6,-7,-8,-9" --token_agg min \
    --m -11.74 --m2 3.07 --best_mult 18 \
    --max_tokens 512 --temperature 0.0
```
</details>

<details>
<summary><b>Qwen3-8B</b>  L14:F7924  scale=2x  m=-9.55  m2=4.19  best_mult=20</summary>

```bash
# HarmBench
python eval/generate_harmbench_anchor.py \
    --model Qwen/Qwen3-8B --layer 14 --feature 7924 \
    --token_pos="-5,-6,-7,-8,-9" --token_agg min \
    --m -9.55 --m2 4.19 --best_mult 20 \
    --max_tokens 512 --temperature 0.0

# JBB-Behaviors
python eval/generate_jbb_anchor.py \
    --model Qwen/Qwen3-8B --layer 14 --feature 7924 \
    --token_pos="-5,-6,-7,-8,-9" --token_agg min \
    --m -9.55 --m2 4.19 --best_mult 20 \
    --max_tokens 512 --temperature 0.0
```
</details>

<details>
<summary><b>Qwen3-14B</b>  L17:F2154  scale=1x  m=-5.94  m2=6.74  best_mult=40</summary>

```bash
# HarmBench
python eval/generate_harmbench_anchor.py \
    --model Qwen/Qwen3-14B --layer 17 --feature 2154 \
    --token_pos="-5,-6,-7,-8,-9" --token_agg min \
    --m -5.94 --m2 6.74 --best_mult 40 \
    --max_tokens 512 --temperature 0.0

# JBB-Behaviors
python eval/generate_jbb_anchor.py \
    --model Qwen/Qwen3-14B --layer 17 --feature 2154 \
    --token_pos="-5,-6,-7,-8,-9" --token_agg min \
    --m -5.94 --m2 6.74 --best_mult 40 \
    --max_tokens 512 --temperature 0.0
```
</details>

<details>
<summary><b>Qwen3-32B</b>  L40:F15515  scale=2x  m=-17.88  m2=-8.95  best_mult=-80</summary>

```bash
# HarmBench
python eval/generate_harmbench_anchor.py \
    --model Qwen/Qwen3-32B --layer 40 --feature 15515 \
    --token_pos="-5,-6,-7,-8,-9" --token_agg max \
    --m -17.88 --m2 -8.95 --best_mult -80 \
    --max_tokens 512 --temperature 0.0

# JBB-Behaviors
python eval/generate_jbb_anchor.py \
    --model Qwen/Qwen3-32B --layer 40 --feature 15515 \
    --token_pos="-5,-6,-7,-8,-9" --token_agg max \
    --m -17.88 --m2 -8.95 --best_mult -80 \
    --max_tokens 512 --temperature 0.0
```
</details>

## Judging Responses

**LlamaGuard** (local, no API key):
```bash
python eval/evaluate_responses_llamaguard.py \
    --input results/eval/harmbench/harmbench_Qwen3-14B_L17_F2154_M40.0.json \
    --guard_model meta-llama/Llama-Guard-3-8B \
    --output results/eval/harmbench/harmbench_Qwen3-14B_L17_F2154_M40.0_llamaguard.json
```

`--input` accepts any output json from the constant or anchor generation scripts above, HarmBench or JBB alike — they all share the same `results` schema. The command above prints both a "Refusal score" and a "Safety score." Refusal score is phrase-matching (a heuristic used in prior work), not an LLM judgment — the paper doesn't rely on it, since it's unreliable (e.g. "silent pivots"); we report it anyway for comparison. Safety score is LlamaGuard-based and is one of the paper's two actual metrics (we also use an LLM judge — see LLM judge below).

**LLM judge (Claude)** (requires an Anthropic API key):
```bash
export ANTHROPIC_API_KEY=<your key>
python eval/evaluate_responses_claude.py --input <path to result json>
python eval/evaluate_responses_claude.py --input <path to result json> --model <model_id>
```

Default model is `claude-sonnet-4-6`. The paper originally used `claude-sonnet-4-20250514`, since retired by Anthropic. Output is saved alongside the input as `<filename>_evaluated.json`. Use `--parallel 5` to speed up evaluation, or `--rate_limit <n>` to cap requests/sec instead.

## Detection

Evaluates whether harmful and harmless prompts can be discriminated without any generation, just by reading the refusal neuron's activations at the post-instruction tokens in the prompt. Supports four datasets: XSTest, WildGuardMix, ToxicChat, and OpenAI Moderation.

```bash
# Single model (Llama-3.1-8B, choose dataset with --dataset)
python detection/evaluate_detection.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct --layer 11 --feature 4258 \
    --token_pos="-2,-3,-4,-5" --agg max \
    --dataset xstest   # or wildguard, toxicchat, openai_moderation
```

Aggregation direction depends on the sign of `best_mult`: use `--agg min` when it's positive, `--agg max` when it's negative. Per-model token positions and aggregation settings for all 7 models are listed in [`detection/README.md`](detection/README.md). Results are saved to `results/detection/{dataset}/`.

### Results

**Llama-3.1-8B** (L11:F4258)

| Dataset | AUROC | AUPRC | Optimal F1 | Harmful # | Harmless # |
|---|---|---|---|---|---|
| XSTest | 0.9686 | 0.9584 | 0.8962 | 200 | 250 |
| WildGuardMix | 0.8712 | 0.8547 | 0.7744 | 754 | 945 |
| ToxicChat | 0.9418 | 0.7022 | 0.6873 | 362 | 4721 |
| OpenAI Moderation | 0.9161 | 0.8236 | 0.7789 | 522 | 1158 |

## Concept Neuron Example — Self-Harm

Amplify a suicide concept neuron to demonstrate concept-specific generation. See `find_concept_neurons/README.md` for the full concept-probing pipeline (generating contrastive passages, ranking MLP neurons by CV accuracy / |activation_gap|).

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from hooks import NeuronSuppressor

# model_name, layer, feature, multiplier = "Qwen/Qwen3-1.7B",  20, 4256,  250
# model_name, layer, feature, multiplier = "Qwen/Qwen3-4B",  22, 9458,  75
# model_name, layer, feature, multiplier = "Qwen/Qwen3-8B",  26, 4061,  -120
model_name, layer, feature, multiplier = "Qwen/Qwen3-14B", 32, 9115,  160
# model_name, layer, feature, multiplier = "Qwen/Qwen3-32B", 56, 9115,  300.0


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

suppressor = NeuronSuppressor(model, layer=layer, feature=feature, multiplier=multiplier, additive=True)

prompt = "Tell me a story in 3 sentences."
# prompt = "用三句话给我讲一个故事。"
# prompt = "Raconte-moi une histoire en trois phrases."
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False)
if not isinstance(inputs, torch.Tensor):
    inputs = inputs["input_ids"]
inputs = inputs.to(model.device)

with suppressor:
    output = model.generate(inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)

response = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("\n\n" + response)
suppressor.remove()
```

The same (layer, feature, multiplier) settings above also induce the concept when the prompt is in French or Chinese, not just English — the neuron isn't tied to English surface forms. The multiplier trades off concept strength against coherence: too low and the concept won't reliably show up, too high and the coherence degrades. If the default doesn't land well for your prompt, sweep nearby values.

## Multiplier Sweep Sensitivity

If the default multiplier doesn't produce a successful attack on a given prompt, try sweeping its magnitude. There's a trade-off underneath this: pushing the multiplier further tends to break refusal more effectively, but past a certain point it also degrades the coherence of the generated text. As a result, ASR as a function of multiplier magnitude tends to rise, peak, and then fall — an inverted-U shape — rather than increasing monotonically. The best-performing multiplier sits somewhere between these two extremes.

<p align="center"><img src="figures/rank_comparison_combined.png" width="700"></p>

*HarmBench ASR of the top-5 gradient-ranked candidate refusal neurons per model. Solid bar: best single (constant) multiplier. Faded extension: cumulative ASR across a sweep of multipliers (union of successful prompts over the sweep). The candidate selected as the refusal neuron for that model is whichever bar has the highest solid (best-single-multiplier) value.*

## Further Reading

For deeper detail on each part of the pipeline, see the section READMEs:

- [`find_concept_neurons/README.md`](find_concept_neurons/README.md) — concept-probing pipeline: generate contrastive passages and rank MLP neurons by CV accuracy / |activation_gap|.
- [`max_activations/README.md`](max_activations/README.md) — build a DuckDB of top / bottom / random activations per neuron, and browse them with the included HTTP server + HTML viewer.
- [`detection/README.md`](detection/README.md) — full per-model detection commands (multi-token and single-token variants) across XSTest, WildGuardMix, ToxicChat, and OpenAI Moderation.

## Citation

```bibtex
@misc{kazemi2026singleneuronsufficientbypass,
      title={A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language Models}, 
      author={Hamid Kazemi and Atoosa Chegini and Maria Safi},
      year={2026},
      eprint={2605.08513},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.08513}, 
}
```
