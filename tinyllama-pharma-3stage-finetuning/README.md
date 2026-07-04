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

### Sample Inference (text continuation)
> _[space reserved for run output — completions for the 3 continuation prompts]_

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

### Sample Inference (Q&A)
> _[space reserved for run output — model responses for the 4 test questions]_

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

### Sample Inference (preference-tuned)
> _[space reserved for run output — model responses for the 4 preference-tuned test questions]_

---

## All Hugging Face Artifacts

| Stage | Adapter | Merged Model |
|---|---|---|
| 1. Non-instruction | [`SivaSai8143/pharma-tinyllama-non-instruction-lora-adapter`](https://huggingface.co/SivaSai8143/pharma-tinyllama-non-instruction-lora-adapter) | [`SivaSai8143/pharma-tinyllama-non-instruction-merged`](https://huggingface.co/SivaSai8143/pharma-tinyllama-non-instruction-merged) |
| 2. Instruction (SFT) | [`SivaSai8143/pharma-tinyllama-instruction-lora-adapter`](https://huggingface.co/SivaSai8143/pharma-tinyllama-instruction-lora-adapter) | [`SivaSai8143/pharma-tinyllama-instruction-merged`](https://huggingface.co/SivaSai8143/pharma-tinyllama-instruction-merged) |
| 3. Preference (DPO) | [`SivaSai8143/pharma-tinyllama-dpo-lora-adapter`](https://huggingface.co/SivaSai8143/pharma-tinyllama-dpo-lora-adapter) | [`SivaSai8143/pharma-tinyllama-dpo-merged`](https://huggingface.co/SivaSai8143/pharma-tinyllama-dpo-merged) |

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




---

## Tech Stack

`transformers` · `peft` · `trl` · `datasets` · `accelerate` · `bitsandbytes` · `pymupdf` · Google Colab (T4 GPU)

---

## Roadmap

- [ ] Add Unsloth implementation of the same 3-stage pipeline
- [ ] Add training loss curves / eval metrics per stage
- [ ] Add before/after qualitative comparison table across all 3 stages
- [ ] Expand corpus and dataset size beyond the demo scale

---

## Disclaimer

This is an educational fine-tuning project for demonstrating LLM training pipelines. The underlying pharma content is for technical demonstration only and is **not medical advice**.
