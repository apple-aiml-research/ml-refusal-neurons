# Detection

Evaluates whether a refusal neuron's activation discriminates harmful from harmless prompts. Supports four datasets:

| Dataset | HuggingFace ID |
|---|---|
| XSTest | `walledai/XSTest` |
| WildGuardMix | `allenai/wildguardmix` |
| ToxicChat | `lmsys/toxic-chat` |
| OpenAI Moderation | `mmathys/openai-moderation-api-evaluation` |

---

## Usage

### Multi-token variant

Uses several token positions and aggregates with `min` or `max`. Aggregation direction depends on the sign of `best_mult`: `--agg min` when it's positive, `--agg max` when it's negative — see each command below for its model's value.

**Qwen3 models** (`--token_pos="-5,-6,-7,-8,-9"`):

<details open>
<summary><b>Qwen3-1.7B</b>  L13:F3270</summary>

```bash
python detection/evaluate_detection.py \
    --model Qwen/Qwen3-1.7B --layer 13 --feature 3270 \
    --token_pos="-5,-6,-7,-8,-9" --agg min --dataset xstest
```
</details>

<details>
<summary><b>Qwen3-4B</b>  L14:F5590</summary>

```bash
python detection/evaluate_detection.py \
    --model Qwen/Qwen3-4B --layer 14 --feature 5590 \
    --token_pos="-5,-6,-7,-8,-9" --agg min --dataset xstest
```
</details>

<details>
<summary><b>Qwen3-8B</b>  L14:F7924</summary>

```bash
python detection/evaluate_detection.py \
    --model Qwen/Qwen3-8B --layer 14 --feature 7924 \
    --token_pos="-5,-6,-7,-8,-9" --agg min --dataset xstest
```
</details>

<details>
<summary><b>Qwen3-14B</b>  L17:F2154</summary>

```bash
python detection/evaluate_detection.py \
    --model Qwen/Qwen3-14B --layer 17 --feature 2154 \
    --token_pos="-5,-6,-7,-8,-9" --agg min --dataset xstest
```
</details>

<details>
<summary><b>Qwen3-32B</b>  L40:F15515</summary>

```bash
python detection/evaluate_detection.py \
    --model Qwen/Qwen3-32B --layer 40 --feature 15515 \
    --token_pos="-5,-6,-7,-8,-9" --agg max --dataset xstest
```
</details>

**Llama models** (`--token_pos="-2,-3,-4,-5"`):

<details open>
<summary><b>Llama-3.1-8B</b>  L11:F4258</summary>

```bash
python detection/evaluate_detection.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct --layer 11 --feature 4258 \
    --token_pos="-2,-3,-4,-5" --agg max --dataset xstest
```
</details>

<details>
<summary><b>Llama-3.1-70B</b>  L25:F10201</summary>

```bash
python detection/evaluate_detection.py \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct --layer 25 --feature 10201 \
    --token_pos="-2,-3,-4,-5" --agg max --dataset xstest
```
</details>

Replace `--dataset xstest` with `wildguard`, `toxicchat`, or `openai_moderation` as needed.

---

### LlamaGuard baseline

Runs LlamaGuard-3-8B as a harmful-prompt classifier on the same four datasets, computing AUROC, AUPRC, and accuracy/F1/precision/recall at LlamaGuard's own native safe/unsafe decision.

```bash
python detection/evaluate_llamaguard.py \
    --guard_model meta-llama/Llama-Guard-3-8B --dataset xstest   # or wildguard, toxicchat, openai_moderation
```

---

### Single-token variant

Uses the single token position identified during neuron selection (from `data/gradient_rankings.json`).

**Qwen3 models:**

<details open>
<summary><b>Qwen3-1.7B</b>  L13:F3270  best_token=-8</summary>

```bash
python detection/evaluate_detection.py \
    --model Qwen/Qwen3-1.7B --layer 13 --feature 3270 \
    --token_pos="-8" --agg min --dataset xstest
```
</details>

<details>
<summary><b>Qwen3-4B</b>  L14:F5590  best_token=-8</summary>

```bash
python detection/evaluate_detection.py \
    --model Qwen/Qwen3-4B --layer 14 --feature 5590 \
    --token_pos="-8" --agg min --dataset xstest
```
</details>

<details>
<summary><b>Qwen3-8B</b>  L14:F7924  best_token=-8</summary>

```bash
python detection/evaluate_detection.py \
    --model Qwen/Qwen3-8B --layer 14 --feature 7924 \
    --token_pos="-8" --agg min --dataset xstest
```
</details>

<details>
<summary><b>Qwen3-14B</b>  L17:F2154  best_token=-6</summary>

```bash
python detection/evaluate_detection.py \
    --model Qwen/Qwen3-14B --layer 17 --feature 2154 \
    --token_pos="-6" --agg min --dataset xstest
```
</details>

<details>
<summary><b>Qwen3-32B</b>  L40:F15515  best_token=-8</summary>

```bash
python detection/evaluate_detection.py \
    --model Qwen/Qwen3-32B --layer 40 --feature 15515 \
    --token_pos="-8" --agg max --dataset xstest
```
</details>

**Llama models:**

<details open>
<summary><b>Llama-3.1-8B</b>  L11:F4258  best_token=-5</summary>

```bash
python detection/evaluate_detection.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct --layer 11 --feature 4258 \
    --token_pos="-5" --agg max --dataset xstest
```
</details>

<details>
<summary><b>Llama-3.1-70B</b>  L25:F10201  best_token=-3</summary>

```bash
python detection/evaluate_detection.py \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct --layer 25 --feature 10201 \
    --token_pos="-3" --agg max --dataset xstest
```
</details>

Running the commands above writes results to `results/detection/{dataset}/` as JSON files containing raw activations (or LlamaGuard scores), metrics, and prompt lists.

---

## Results

**Llama-3.1-8B refusal neuron (L11:F4258) vs. LlamaGuard-3-8B baseline** — same prompts/splits for both; cells show `neuron / LlamaGuard`.

LlamaGuard's F1/Precision/Recall are computed at its native decision boundary — the "safe"/"unsafe" token it actually generates (`unsafe_score > 0.5`). The neuron has no native decision boundary of its own, so its F1/Precision/Recall are computed at the threshold that maximizes F1. AUROC/AUPRC are threshold-independent and computed identically for both.

| Dataset | AUROC | AUPRC | F1 | Precision | Recall | Harmful # | Harmless # |
|---|---|---|---|---|---|---|---|
| XSTest | 0.9686 / 0.9753 | 0.9584 / 0.9691 | 0.8962 / 0.8817 | 0.8482 / 0.9535 | 0.9500 / 0.8200 | 200 | 250 |
| WildGuardMix | 0.8712 / 0.9159 | 0.8547 / 0.9117 | 0.7744 / 0.7676 | 0.7340 / 0.9318 | 0.8196 / 0.6525 | 754 | 945 |
| ToxicChat | 0.9418 / 0.8713 | 0.7022 / 0.4849 | 0.6873 / 0.4887 | 0.7011 / 0.4729 | 0.6740 / 0.5055 | 362 | 4721 |
| OpenAI Moderation | 0.9161 / 0.9218 | 0.8236 / 0.8727 | 0.7789 / 0.7877 | 0.7311 / 0.7900 | 0.8333 / 0.7854 | 522 | 1158 |

---

## Note

In our experiments, ensembling the refusal-neuron activations of two different models (z-score normalizing and averaging per-prompt scores before computing AUROC/AUPRC/F1) improved detection over either model alone. This is off-thesis for the paper — the paper's claim is about a *single* neuron in a *single* model — so we did not pursue it further and no ensembling code is included in this repository.
