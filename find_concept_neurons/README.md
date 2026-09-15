# Finding Concept Neurons

> [!CAUTION]
> This section may contain or generate harmful content, including self-harm and suicide, used strictly for research and evaluation purposes.

Identifies individual MLP neurons that encode a specific concept (e.g., "suicide") by:

1. Generating contrastive positive/negative example passages with a public uncensored HuggingFace model.
2. For every MLP neuron in every layer, fitting a 1-feature cross-validated logistic regression on its activation to score how well it separates positives from negatives.
3. Ranking neurons per-layer and globally by CV accuracy, absolute activation gap, and related magnitude metrics.

---

## Quick Start

Amplify a suicide concept neuron to demonstrate concept-specific generation:

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

---

## Step 1 — Generate examples

Uses the generation pipeline in [`generation/`](generation). Deterministic given a seed; runs entirely on public HuggingFace models. `huihui-ai/Qwen3-32B-abliterated` is a safety-training-removed variant used only because aligned models refuse to produce the positive class needed for contrastive probing.

We do **not** ship the generated examples with this repository. The generator model is gated on HuggingFace, and users need to accept its terms before pulling it — so the data must be regenerated locally. The command below shows how to generate 600 passages (300 positive + 300 negative) for a given concept.

```bash
cd find_concept_neurons/generation
python scripts/generate_examples.py \
    --concept suicide \
    --k 300 \
    --batch_size 50 \
    --local \
    --local_model huihui-ai/Qwen3-32B-abliterated \
    --seed 42
```

Writes `examples_data/examples_suicide.json` with 300 positive + 300 negative passages. Move this to `find_concept_neurons/examples_data/examples_suicide_seed42.json` before running Step 2, or point `--examples_file` at wherever you saved it.

---

## Step 2 — Find the concept neuron

Ranks every MLP neuron in the target model by CV accuracy and absolute activation gap. `--target L:F` points to the neuron reported in the paper for that model. Output JSON contains per-layer top features and a global top-100 ranking.

To inspect the token spans that maximally/minimally activate each candidate neuron, see the [`max_activations/`](../max_activations) folder.

<details open>
<summary><b>Qwen3-1.7B</b></summary>

```bash
CUDA_VISIBLE_DEVICES=0 python find_concept_neurons/find_concept_neurons.py \
    --model Qwen/Qwen3-1.7B --concept "suicide" \
    --token_aggregation max \
    --rankings activation_gap cv \
    --target 20:4256 \
    --examples_file find_concept_neurons/examples_data/examples_suicide_seed42.json \
    --output results/find_concept_neurons/features_Qwen3-1.7B_suicide.json
```
</details>

<details>
<summary><b>Qwen3-4B</b></summary>

```bash
CUDA_VISIBLE_DEVICES=0 python find_concept_neurons/find_concept_neurons.py \
    --model Qwen/Qwen3-4B --concept "suicide" \
    --token_aggregation max \
    --rankings activation_gap cv \
    --target 22:9458 \
    --examples_file find_concept_neurons/examples_data/examples_suicide_seed42.json \
    --output results/find_concept_neurons/features_Qwen3-4B_suicide.json
```
</details>

<details>
<summary><b>Qwen3-8B</b></summary>

```bash
CUDA_VISIBLE_DEVICES=0 python find_concept_neurons/find_concept_neurons.py \
    --model Qwen/Qwen3-8B --concept "suicide" \
    --token_aggregation min \
    --rankings activation_gap cv \
    --target 26:4061 \
    --examples_file find_concept_neurons/examples_data/examples_suicide_seed42.json \
    --output results/find_concept_neurons/features_Qwen3-8B_suicide.json
```
</details>

<details>
<summary><b>Qwen3-14B</b></summary>

```bash
CUDA_VISIBLE_DEVICES=0 python find_concept_neurons/find_concept_neurons.py \
    --model Qwen/Qwen3-14B --concept "suicide" \
    --token_aggregation max \
    --rankings activation_gap cv \
    --target 32:9115 \
    --examples_file find_concept_neurons/examples_data/examples_suicide_seed42.json \
    --output results/find_concept_neurons/features_Qwen3-14B_suicide.json
```
</details>

<details>
<summary><b>Qwen3-32B</b></summary>

```bash
CUDA_VISIBLE_DEVICES=0 python find_concept_neurons/find_concept_neurons.py \
    --model Qwen/Qwen3-32B --concept "suicide" \
    --token_aggregation max \
    --rankings activation_gap cv \
    --target 56:9115 \
    --examples_file find_concept_neurons/examples_data/examples_suicide_seed42.json \
    --output results/find_concept_neurons/features_Qwen3-32B_suicide.json
```
</details>

---

## What the metrics mean

Each neuron gets one score per metric. The metrics are all computed from the same collected activations, so this is a single-pass scoring — no additional forward passes for extra metrics.

- **CV accuracy** — For every MLP neuron, we take that neuron's activation across all examples and fit a 1-feature logistic regression on it using stratified k-fold cross-validation. The score is the mean held-out validation accuracy. A neuron with CV = 1.0 perfectly separates positives from negatives using its own activation alone; the model has effectively dedicated that neuron to the concept.

- **|activation_gap|** (the `activation_gap` metric in `--rankings`) — The absolute difference between the mean activation on positives and the mean activation on negatives: `|mean(h_act) − mean(n_act)|`. A high gap means the neuron reliably shifts in magnitude between the two classes. No classifier needed — it's a direct summary statistic.

- **|activation_gap_q1|**, **|activation_gap_median|** — Same idea as |activation_gap| but computed on the first quartile or median instead of the mean. More robust to outlier activations, which matters when a small handful of examples produce very large activations that skew the mean.

- **combined_q1**, **combined_median** — `|activation_gap_q1| × (1 − bimodality) × selectivity` (or median version). Combines contrast magnitude with two additional signals: a *bimodality* penalty (prefer unimodal within-class distributions) and a *selectivity* bonus (prefer neurons that fire only for the target class, not everything).

CV and contrast capture different things and often disagree at the top. CV rewards *separability* — a neuron whose activation clearly ranks positives above negatives, even if the magnitude gap is small. Contrast rewards *magnitude*. A neuron can have CV = 1.0 (perfect linear separator) with tiny magnitude difference, or huge magnitude difference with imperfect ranking (some overlap). Combining them usually finds the neuron the model genuinely dedicated to the concept. **We use `activation_gap + cv` for all the experiments in this section and the commands above.**

---

## Key flags for `find_concept_neurons.py`

| Flag | What it controls |
|---|---|
| `--rankings` | Metrics to include in the average rank, in order (first is the tie-breaker on `avg_rank`). Default: `activation_gap cv`. Choices: `cv`, `activation_gap`, `q1`, `median`, `combined_q1`, `combined_median`. |
| `--token_aggregation` | How to reduce per-token activations to one scalar per example: `max`, `min`, or `mean` (default: `max`). |
| `--target L:F` | Highlight the neuron at layer `L` feature `F` wherever it appears in the ranking tables, and always report its rank — even when it falls below the rows shown. |
| `--layers` | Comma-separated layers to analyze (default: all). |
| `--top_features` | Number of top features to keep per layer (default: 10). |
| `--cv_folds` | Number of CV folds for the per-neuron logistic regression (default: 2). |
| `--capture_stage` | Which MLP stage to probe: `gated` (default, = `act_fn(gate_proj(x)) * up_proj(x)` — the SwiGLU gated activation), `up_proj`, `gate_proj`, `down_proj`, or `residual`. |
