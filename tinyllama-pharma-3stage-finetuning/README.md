# Pharma-Domain LLM Fine-Tuning: Non-Instruction → Instruction → Preference (DPO)

End-to-end pipeline that takes a single domain PDF (pharmacology/pharma R&D corpus) and progressively specialises a base LLM through **three sequential LoRA fine-tuning stages**, each stage building on the merged output of the previous one.

This repo is a hands-on demonstration of the **continued-pretraining → SFT → DPO** workflow used in real instruction-tuned model pipelines, implemented twice:

- `huggingface/` — Transformers + PEFT + TRL (Google Colab)
- `unsloth/` — Unsloth-accelerated version (Google Colab)

---

## Pipeline Overview

```text
Pharma PDF (raw domain text)
        │
        ▼
┌───────────────────────────────────────────┐
│ Stage 1 — Non-Instruction Fine-Tuning      │
│ (Continued Pretraining / Causal LM on      │
│  raw pharma text)                          │
│  Base model: TinyLlama-1.1B + LoRA (QLoRA) │
└───────────────────────────────────────────┘
        │  save adapter → merge adapter into base
        ▼
   Merged Stage-1 Model
        │  load as new base, attach fresh LoRA
        ▼
┌───────────────────────────────────────────┐
│ Stage 2 — Instruction Fine-Tuning (SFT)   │
│ Alpaca-style instruction/response data    │
└───────────────────────────────────────────┘
        │  save adapter → merge adapter into base
        ▼
   Merged Stage-2 Model
        │  load as new base, attach fresh LoRA
        ▼
┌───────────────────────────────────────────┐
│ Stage 3 — Preference Tuning (DPO)         │
│ prompt / chosen / rejected pairs          │
└───────────────────────────────────────────┘
        │  save adapter → merge adapter into base
        ▼
   Final Merged Preference-Tuned Model
```

Each stage produces **three artifacts**: the LoRA adapter, the tokeniser, and the full merged model (base + adapter) — and the merged model from stage *N* becomes the base model loaded for stage *N+1*.

---

## Why This Project

Most public LoRA fine-tuning tutorials demonstrate a single stage in isolation. This project chains all three stages that make up a realistic alignment pipeline:

| Stage | Data format | What the model learns |
|---|---|---|
| 1. Non-instruction FT | Raw text | Domain vocabulary, terminology, writing style |
| 2. Instruction FT (SFT) | `instruction → response` | How to follow instructions and answer questions |
| 3. Preference Tuning (DPO) | `prompt → chosen vs rejected` | To prefer higher-quality answers over plausible-but-wrong ones |

All three stages are trained on the **same small pharma corpus** (Metformin pharmacology, statin/ezetimibe lipid-lowering therapy, mRNA vaccines, AI in drug discovery, and clinical trial/pharmacovigilance terminology), so the model's progression can be directly observed at each stage on the same domain.

---

## Repository Structure

```text
.
├── data/
│   ├── Metformin-Lipid-Therapy-Knowledge.pdf   # source domain corpus
│   ├── pharma_paragraph_process.jsonl          # cleaned paragraphs (Stage 1 input)
│   ├── pharma_instruction_dataset.jsonl        # instruction/response pairs (Stage 2)
│   └── pharma_preference_dataset.jsonl         # prompt/chosen/rejected pairs (Stage 3)
├── huggingface/
│   ├── 1_Non_Instruction_Causal_LLM_Fine_Tuning_or_Domain_Adaptive_Continued_Pretraining.ipynb     # full HF + PEFT + TRL pipeline (Colab)
│   ├── 2_Non_Instruction_Causal_LLM_+_Instruction_Fine_Tuning.ipynb  
│   └── 3_Non_Instruction_Causal_LLM_+_Instruction_+_Preference_Tuning_with_DPO.ipynb   
├── unsloth/
│   └── End-to-End-LLM-Finetuning-with-Unsloth.ipynb
└── README.md
```

---

## Base Model & Common Setup

- **Base model:** [`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
- **Method:** LoRA / QLoRA (4-bit `nf4` quantization via `bitsandbytes`, fp16 compute)
- **Frameworks:** `transformers`, `peft`, `datasets`, `trl`, `accelerate`, `pymupdf`
- **Environment:** Google Colab (T4 GPU)

**LoRA target modules (all stages):**
```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

---

## Stage 1 — Non-Instruction Fine-Tuning (Continued Pretraining)

**Goal:** Adapt the base model to pharma-domain language via causal next-token prediction on raw text — no instruction format involved.

### Data Pipeline
1. Extract text per page from the PDF with PyMuPDF
2. Clean & normalise (Unicode NFKC normalisation, de-hyphenation, whitespace/page-number cleanup)
3. Split into paragraph-level records, dropping paragraphs under `min_chars_per_paragraph`
4. Convert to a Hugging Face `Dataset`, train/validation split
5. Tokenise (no padding), then pack tokens into fixed-size `block_size` blocks for efficient causal LM training (labels = input_ids)

### Config
| Parameter | Value |
|---|---|
| `block_size` | 512 |
| `min_chars_per_paragraph` | 80 |
| `test_size` | 0.15 |
| `seed` | 42 |
| `lora_r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `num_train_epochs` | 3.0 |
| `max_steps` | -1 (full epochs) |
| `per_device_train_batch_size` | 1 |
| `per_device_eval_batch_size` | 1 |
| `gradient_accumulation_steps` | 8 |
| `learning_rate` | 2e-4 |
| `warmup_ratio` | 0.03 |
| `weight_decay` | 0.01 |
| `eval_steps` | 10 |
| `save_steps` | 25 |
| `save_total_limit` | 2 |

### Training
- Quantised 4-bit base model (`BitsAndBytesConfig`, `nf4`, double quant) prepared via `prepare_model_for_kbit_training`
- `DataCollatorForLanguageModeling(mlm=False)`
- Trained with `transformers.Trainer`

### Output Artifacts
- LoRA adapter → `SivaSai8143/pharma-tinyllama-non-instruction-lora-adapter`
- Merged model → `SivaSai8143/pharma-tinyllama-non-instruction-merged`

---

## Stage 2 — Instruction Fine-Tuning (SFT)

**Goal:** Teach the Stage-1 domain-adapted model to follow instructions and answer pharma questions in a structured Q&A format.

### Data
`pharma_instruction_dataset.jsonl` — 48 instruction/response pairs covering:

- Metformin pharmacology, pharmacokinetics, safety, clinical use & endpoints
- Lipid-lowering therapy (atorvastatin + ezetimibe), familial hypercholesterolemia
- mRNA vaccine platforms and immune response
- AI in drug discovery, lead optimisation, ADME/toxicology, regulatory AI
- Clinical trial terminology and pharmacovigilance

**Example record:**
```json
{
  "instruction": "Explain the primary mechanism of action of metformin.",
  "input": "",
  "output": "Metformin primarily acts by activating AMP-activated protein kinase, also called AMPK. AMPK is a central metabolic regulator that promotes glucose uptake and fatty acid oxidation while reducing hepatic gluconeogenesis, which helps lower blood glucose levels.",
  "source_page": 1,
  "topic": "Metformin pharmacology"
}
```

Each record is formatted into an Alpaca-style prompt:
```text
### Instruction:
<instruction>

### Response:
<output>
```

### Approach
- **Base for this stage = merged Stage-1 model** (loaded in 4-bit), with a **new LoRA adapter** attached (continuing as a fresh adapter on top of the domain-adapted merged weights, rather than continuing to train the Stage-1 adapter directly)
- Tokenised with `max_length=512`, padding to max length, labels masked with `-100` on padding tokens
- `DataCollatorForLanguageModeling(mlm=False)`, trained with `transformers.Trainer`

### LoRA Config (Stage 2)
| Parameter | Value |
|---|---|
| `r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| target modules | q/k/v/o_proj, gate/up/down_proj |

### Training Args (Stage 2)
| Parameter | Value |
|---|---|
| `num_train_epochs` | 5 |
| `max_steps` | 5 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 8 |
| `learning_rate` | 1e-4 |
| `warmup_steps` | 2 |
| `weight_decay` | 0.01 |
| eval strategy | every step |
| train/validation split | 85% / 15% (seed 42) |

### Output Artifacts
- LoRA adapter → `SivaSai8143/pharma-tinyllama-instruction-lora-adapter`
- Merged model → `SivaSai8143/pharma-tinyllama-instruction-merged`

---

## Stage 3 — Preference Tuning with DPO

**Goal:** Align the instruction-tuned model toward higher-quality answers using Direct Preference Optimisation ([Rafailov et al., 2023](https://arxiv.org/pdf/2305.18290)).

### Data
`pharma_preference_dataset.jsonl` — 48 `prompt / chosen/rejected` triples, built on the same instruction prompts as Stage 2. `chosen` responses are the accurate domain answers; `rejected` responses are plausible-sounding but factually wrong or off-target answers.

**Example record:**
```json
{
  "prompt": "### Instruction:\nExplain the primary mechanism of action of metformin.\n\n### Response:\n",
  "chosen": "Metformin primarily acts by activating AMP-activated protein kinase, also called AMPK. AMPK is a central metabolic regulator that promotes glucose uptake and fatty acid oxidation while reducing hepatic gluconeogenesis, which helps lower blood glucose levels.",
  "rejected": "Metformin mainly works by increasing insulin secretion from the pancreas, and kidney function is usually not very relevant. Its side effects are generally not important unless the patient feels very sick.",
  "source_page": 1,
  "topic": "Metformin pharmacology"
}
```

### Approach
- **Base for this stage = merged Stage-2 (instruction-tuned) model** (loaded in 4-bit), with a **new LoRA adapter** for preference tuning
- Trained with `trl.DPOTrainer` + `trl.DPOConfig`, `ref_model=None` (TRL handles the reference policy internally)
- Train/validation split: 85% / 15% (seed 42)

### DPO Config
| Parameter | Value |
|---|---|
| `beta` | 0.1 |
| `max_length` | 512 |
| `max_prompt_length` | 256 |
| `num_train_epochs` | 3 |
| `max_steps` | 5 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 8 |
| `learning_rate` | 5e-5 |
| `warmup_steps` | 2 |
| `weight_decay` | 0.01 |
| eval strategy | every step |

### Output Artifacts
- LoRA adapter → `SivaSai8143/pharma-tinyllama-dpo-lora-adapter`
- Final merged model → `SivaSai8143/pharma-tinyllama-dpo-merged`

---

## Evaluation Strategy

### Why separate metrics per stage?
 
Each fine-tuning stage has a fundamentally different objective — continued pretraining teaches domain language, SFT teaches instruction-following, and DPO teaches preference alignment. A single metric cannot capture all three. Using the right metric per stage shows the model is actually learning what each stage intends to teach, rather than optimising a proxy signal that happens to correlate with training loss.
 
---

### Stage 1 — Perplexity
 
**What it is:**
Perplexity measures how confidently the model predicts the next token on unseen text. It is computed as `e ^ validation_loss`, where validation_loss is the average cross-entropy loss over the held-out pharma paragraphs. Intuitively — the lower the perplexity, the less "surprised" the model is by pharma text.
 
**Why this metric for this stage:**
Stage 1 is continued pretraining — pure next-token prediction on raw domain text. There are no instructions, no reference answers, and no preference pairs to compare against. Perplexity is the natural and standard metric for this objective because it directly measures how well the model has internalised the statistical patterns of the domain corpus. Comparing base model perplexity vs Stage-1 perplexity on the same held-out pharma sentences gives a clean before/after signal of domain adaptation.

**How to interpret the score:**
Lower perplexity = better domain adaptation.
 
| Perplexity Range | Meaning |
|---|---|
| 20 – 50 | Base model — pharma text is largely unfamiliar |
| 10 – 20 | Partial domain adaptation |
| 5 – 10 | Strong domain adaptation |
| < 5 | Near-perfect fit (risk of overfitting on small corpus) |
 
A meaningful drop from base model perplexity to Stage-1 perplexity on the same pharma validation set confirms the model has absorbed domain vocabulary, terminology, and writing style.
 
**Limitations:**
Perplexity measures fluency and domain fit, not factual correctness or instruction-following ability. A model with low perplexity on pharma text can still generate plausible-sounding but factually wrong statements. This is why Stage 2 evaluation shifts to answer quality metrics.

---
 
### Stage 2 — ROUGE-L + BERTScore
 
**What they are:**

**ROUGE-L** (Recall-Oriented Understudy for Gisting Evaluation — Longest Common Subsequence) measures the longest sequence of tokens that appears in both the generated answer and the reference answer, in the same order, without requiring them to be contiguous. It rewards correct pharma terminology and phrase-level matches while being more flexible than exact n-gram overlap.
 
- Score range: 0.0 to 1.0 — higher is better
- 0.0 = no overlap with reference, 1.0 = exact match
**BERTScore** measures semantic similarity between the generated answer and the reference answer using contextual embeddings from a pretrained BERT model. Unlike ROUGE, it understands meaning — it rewards correct paraphrases even when the exact words differ. We report the F1 score, which balances precision (how much of the generation matches the reference) and recall (how much of the reference is covered by the generation).
 
- Score range: 0.0 to 1.0 — higher is better
- Typical range for good generations: 0.85+
- Scores below 0.70 suggest the generation is semantically distant from the reference
**Why these metrics for this stage:**
Stage 2 is instruction fine-tuning — the model must generate structured answers to pharma questions. We need metrics that evaluate answer quality, not just next-token prediction. ROUGE-L catches whether the model uses the correct pharma terminology and phrase structure. BERTScore catches whether the model is saying the right thing semantically, even if worded differently. Together they cover both surface form and meaning — neither alone is sufficient for evaluating open-ended generation.
 
ROUGE-1 and ROUGE-2 (unigram and bigram overlap) are intentionally excluded — they add noise without adding insight for open-ended generation where paraphrasing is natural and expected.
 
**How to interpret the scores:**
 
| Metric | Range | Interpretation |
|---|---|---|
| ROUGE-L | 0.0 – 0.2 | Low lexical overlap — model paraphrases heavily or misses key terms |
| ROUGE-L | 0.2 – 0.5 | Moderate overlap — model uses some correct terminology |
| ROUGE-L | 0.5+ | Strong overlap — model closely matches reference phrasing |
| BERTScore F1 | < 0.70 | Semantically distant from reference |
| BERTScore F1 | 0.70 – 0.85 | Semantically related — model understands the domain |
| BERTScore F1 | 0.85+ | Semantically close — strong instruction-following quality |
 
On a demo-scale dataset (48 samples, max_steps=5), ROUGE-L will naturally be low due to paraphrase penalty and limited training — this is expected and does not indicate model failure. BERTScore is the more meaningful signal at this scale.
 
**Limitations:**
Both metrics require reference answers — they measure how close the model is to a specific ground truth, not absolute factual correctness. A model can score low on ROUGE-L while still generating a factually accurate answer worded differently. These metrics are directional signals at demo scale, not definitive benchmarks.

---

## All Hugging Face Artifacts

| Stage | Adapter | Merged Model |
|---|---|---|
| 1. Non-instruction | [`SivaSai8143/pharma-tinyllama-non-instruction-lora-adapter`](https://huggingface.co/SivaSai8143/pharma-tinyllama-non-instruction-lora-adapter) | [`SivaSai8143/pharma-tinyllama-non-instruction-merged`](https://huggingface.co/SivaSai8143/pharma-tinyllama-non-instruction-merged) |
| 2. Instruction (SFT) | [`SivaSai8143/pharma-tinyllama-instruction-lora-adapter`](https://huggingface.co/SivaSai8143/pharma-tinyllama-instruction-lora-adapter) | [`SivaSai8143/pharma-tinyllama-instruction-merged`](https://huggingface.co/SivaSai8143/pharma-tinyllama-instruction-merged) |
| 3. Preference (DPO) | [`SivaSai8143/pharma-tinyllama-dpo-lora-adapter`](https://huggingface.co/SivaSai8143/pharma-tinyllama-dpo-lora-adapter) | [`SivaSai8143/pharma-tinyllama-dpo-merged`](https://huggingface.co/SivaSai8143/pharma-tinyllama-dpo-merged) |

---
### Stage 3 — Reward Metrics + Win Rate
 
**What they are:**
 
**Reward Metrics** are logged internally by `DPOTrainer` during training. DPO works by assigning implicit reward scores to chosen and rejected responses — these metrics track how well the training is separating them.
 
**rewards/chosen:** The mean implicit reward the model assigns to the correct (chosen) responses. A positive and increasing value means the model is learning to score accurate answers higher.
 
**rewards/rejected:** The mean implicit reward the model assigns to the plausible-but-wrong (rejected) responses. A negative and decreasing value means the model is learning to down-score incorrect answers.
 
**rewards/margins:** The difference between rewards/chosen and rewards/rejected (chosen − rejected). This is the most direct training signal — a positive and widening margin confirms DPO is successfully separating good answers from bad ones. A margin near zero means the model treats chosen and rejected as equally likely.
 
**rewards/accuracies:** The percentage of training examples where the model correctly ranks the chosen response higher than the rejected response.
 
| Accuracy | Meaning |
|---|---|
| 0.50 | Random — DPO had no effect |
| 0.60 – 0.70 | Moderate alignment |
| 0.70+ | Strong alignment — model reliably prefers correct answers |
| 1.00 | Perfect on training set (expected on very small datasets) |
 
**Win Rate** is a post-training inference evaluation. For each prompt in the test set the model generates a free-form response, which is then compared against both the chosen and rejected references using BERTScore F1. The model wins if its generated response is semantically closer to the chosen (correct) answer than to the rejected (wrong) answer.
 
- Win Rate = number of wins / total test samples
- 50% = random baseline (no preference alignment)
- 60 – 70% = moderate alignment
- 70%+ = strong alignment — DPO meaningfully improved answer quality
**Why these metrics for this stage:**
Stage 3 is preference alignment — the goal is not to generate text that matches a reference, but to steer the model toward preferring accurate answers over plausible-but-wrong ones. ROUGE and BERTScore against a single reference cannot capture this. Reward metrics measure whether DPO training converged correctly (training signal). Win rate measures whether the final model actually generalises that preference to unseen prompts (real-world signal). Both are needed — a model can show healthy reward margins during training but still fail to generalise, which win rate catches.
 
**How to interpret the scores:**
 
| Metric | Good signal | Strong signal |
|---|---|---|
| rewards/chosen | Positive and increasing | > 0 at end of training |
| rewards/rejected | Negative and decreasing | < 0 at end of training |
| rewards/margins | Positive and widening | > 0.3 |
| rewards/accuracies | > 0.60 | > 0.70 |
| Win Rate | > 0.60 | > 0.70 |
 
**Limitations:**
Reward metrics reflect training-set behaviour only — they do not measure factual accuracy or generalisation. Win rate on a small test set (~7 samples after an 85/15 split of 48 examples) has high variance and should be treated as a directional signal. A larger dataset would give statistically robust results.

---
 
### Known Limitations
 
- **Demo-scale dataset** — 48 samples each for instruction and preference stages. All evaluation metrics are directional signals, not production benchmarks. Results should not be compared directly against published fine-tuning papers.
- **max_steps=5 for SFT and DPO** — intentionally limited for reproducibility on a free Colab T4 GPU within session time limits. This is not representative of full fine-tuning convergence.
- **Small test set** — after an 85/15 train/validation split of 48 samples, the test set contains approximately 7 samples. Win rate and BERTScore at this scale have high variance.
- **Single domain, single PDF** — the pharma corpus covers 6 topics from one document. The model's domain adaptation is narrow and should not be generalised beyond this corpus.
---
 
### What Would Improve Results
 
- **Larger dataset** — 1,000–10,000 instruction/preference pairs would give statistically meaningful metrics and meaningfully better model behaviour
- **Full epoch training** — removing `max_steps=5` and training to convergence (5–10 epochs for SFT, 3–5 for DPO) on a larger dataset
- **Larger base model** — Llama-3-8B or Mistral-7B would bring stronger priors and better instruction-following out of the box
- **Broader corpus** — expanding beyond a single PDF to a curated pharma corpus (PubMed abstracts, clinical guidelines, drug monographs)
- **Standardised benchmarks** — evaluating against domain-specific benchmarks such as PubMedQA or MedQA via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) would allow direct comparison with published models
- **Reward model evaluation** — using a trained reward model (rather than BERTScore) for win rate computation gives a more reliable preference signal

---

## Key Engineering Decisions

- **QLoRA (4-bit) throughout** — keeps every stage trainable on a single Colab T4 GPU
- **Adapter saved + base merged at every stage** — produces a clean standalone model that becomes the base for the next stage, avoiding adapter-stacking complexity
- **Same corpus across all 3 stages** — isolates the effect of each fine-tuning technique on identical domain knowledge, making before/after comparisons meaningful
- **`-100` label masking on padding** — ensures loss is computed only on real tokens during instruction tuning
- **DPO with `ref_model=None`** — lets TRL manage the reference policy automatically, simplifying the preference-tuning setup

---

## Key Learnings:

### 1. What does packing=True do in SFT training?
"packing=True does not pack the entire dataset at once."
##### Instead, it:
*   Dynamically combines multiple short samples
*   Packs them into sequences up to max_seq_length (e.g., 512 tokens)
##### Example:
> Sample A (120 tokens)
> + Sample B (200 tokens)
> + Sample C (80 tokens)
> = One 400-token training sequence

It improves GPU utilization and reduces padding waste.

### 2. What is the difference between RIGHT and LEFT padding?<br>
#### 🔹 1. RIGHT Padding (Used in SFT / Stage 1 & 2)
##### Format:
> [TEXT TEXT TEXT TEXT PAD PAD PAD]
##### Example:


> Instruction + Response:
"Explain metformin → Metformin is a drug"

> After tokenization:
[Explain metformin Metformin is a drug PAD PAD PAD]

##### Why RIGHT padding is used:


*   Best for causal language modeling (next-token prediction)
*   Keeps real text at the beginning
*   Padding is ignored naturally by attention
*   Stable for training single sequences

##### Used in:
*   Stage 1 (domain pretraining)
*   Stage 2 (instruction SFT)

#### 🔹 2. LEFT Padding (Used in DPO / Stage 3)
##### Format:

> [PAD PAD PAD TEXT TEXT TEXT TEXT]
##### Example:


> Prompt + Response:
"Explain metformin → Metformin is a drug"

> After tokenization:
[PAD PAD PAD Explain metformin Metformin is a drug]

##### Why LEFT padding is used:

*   To compare two sequences token-by-token under the same prompt alignment”
*   Ensures prompt ends at same position across samples
*   Aligns chosen vs rejected sequences properly
*   Required for correct probability comparison in DPO
*   Makes batching stable for pairwise training

##### Used in:
*   Stage 3 (DPO / preference tuning)

### 3. Can I change “Instruction” to “Prompt” in my format?
Yes.

The model does not care about wording like:
*   Instruction
*   Prompt
*   Question
It only learns structure:
> Input → Output mapping

### 4. How did i handle gradient explosion (loss → NaN) during training?
During training, I encountered gradient explosion, where the loss became NaN due to unstable and excessively large gradient updates.

To fix this, I:
*   Reduced the learning rate to 5e-5
*   Applied gradient clipping using max_grad_norm = 0.1
These changes stabilized training and prevented unstable updates.

##### What is Learning Rate?
The learning rate controls how big a step the model takes while updating its weights during training.
*   High learning rate → large updates → faster but unstable training (can cause NaN loss)
*   Low learning rate → smaller updates → slower but more stable training
In this case, reducing it to 5e-5 helped prevent the model from “overshooting” optimal weights.

##### What is max_grad_norm (Gradient Clipping)?
max_grad_norm limits the size of gradients during backpropagation.
*   If gradients become too large, they are scaled down (clipped)
*   This prevents sudden extreme updates to model weights

##### Example:
> Before clipping: gradient = 50 (too large ❌)
> After clipping:  gradient = 0.1 (safe ✔)

Setting max_grad_norm = 0.1 ensured training remained stable by preventing gradient explosion.
##### Final Result
After applying these fixes:

*   Training stabilized
*   No more NaN loss
*   Final validation loss improved to ~16.8
---

## Tech Stack

`transformers` · `peft` · `trl` · `datasets` · `accelerate` · `bitsandbytes` · `pymupdf`. `unsloth` · Google Colab (T4 GPU)

---

## Roadmap

- [ ] Add Unsloth implementation of the same 3-stage pipeline
- [ ] Add training loss curves / eval metrics per stage
- [ ] Add before/after qualitative comparison table across all 3 stages
- [ ] Expand corpus and dataset size beyond the demo scale

---

## Disclaimer

This is an educational fine-tuning project for demonstrating LLM training pipelines. The underlying pharma content is for technical demonstration only and is **not medical advice**.
