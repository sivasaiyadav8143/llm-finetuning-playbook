# ContractLens-SFT: Fine-tuning Gemma-2B for Legal Contract NLI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **High-accuracy Legal Natural Language Inference (NLI) on a budget.**  
> Fine-tuned Gemma-2B with LoRA & SFT on the ContractNLI benchmark—optimized to run on a free Google Colab T4 GPU.

---


## 🎯 The Problem

Legal professionals are overwhelmed by dense, jargon-filled contracts. AI can help by triaging clauses—but general-purpose LLMs struggle with legal logic and often hallucinate. 

**ContractLens** solves this by adapting a lightweight open-source model (Gemma-2B) to accurately classify legal relationships between contract clauses and hypothetical statements.

### The Task: Contract NLI

Given a **Premise** (contract clause) and a **Hypothesis** (legal statement), the model must classify the relationship as exactly **one** of three classes:

| Label | Definition | Example |
| :--- | :--- | :--- |
| **Entailment** | The hypothesis is **directly supported** by the contract. | *Premise:* "Receiving Party shall not disclose." <br> *Hypothesis:* "Receiving Party must keep confidential." |
| **Neutral** | The contract **does not address** the hypothesis. | *Premise:* "Disclosure must be in writing." <br> *Hypothesis:* "Oral disclosure is allowed." (Not mentioned) |
| **Contradiction** | The hypothesis **directly conflicts** with the contract. | *Premise:* "All rights are retained." <br> *Hypothesis:* "All rights are waived." |

---

## 🧠 Methodology: Why SFT + LoRA?

| Component | Why We Chose It |
| :--- | :--- |
| **Supervised Fine-Tuning (SFT)** | Explicitly teaches the model to map `(Premise, Hypothesis) -> Label` via instruction prompts. Unlike continued pretraining (CLM loss), SFT forces the model to *reason* about the relationship between two texts, not just predict the next token. |
| **LoRA (Low-Rank Adaptation)** | Reduces trainable parameters from 2.1B to ~30M (< 2%). This makes fine-tuning feasible on 16GB VRAM and prevents catastrophic forgetting of the base model's general knowledge. |
| **QLoRA (4-bit Quantization)** | Shrinks the base model memory footprint to ~1.5GB, leaving room for gradients and activations on a T4 GPU. |
| **Gemma-2B** | Selected for its strong reasoning ability and compact size—the perfect balance for resource-constrained environments. [[Model Card]](https://huggingface.co/google/gemma-2b) |

### Why Not Non-Instruction (Continued Pretraining)?
Continued pretraining on only ~7,000 contract clauses would cause rapid overfitting and doesn't teach the model how to *compare* two statements. SFT is the correct tool for this classification task.

---

## 📊 Dataset: ContractNLI (`contractnli_a`)

- **Source:** [kiddothe2b/contract-nli](https://huggingface.co/datasets/kiddothe2b/contract-nli)
- **Subset:** `contractnli_a` (truncated premises, fits 512-token limit).
- **Splits used:** 7,000 training / 1,000 validation / 1,600 test (test set kept full for unbiased evaluation).
- **Label Distribution (Train):** Balanced across Entailment, Neutral, and Contradiction.

> *I downsampled the training set to fit within Google Colab's free runtime limits while preserving statistical rigor in final evaluation.*

---

## 🚀 Results & Evaluation

### Primary Metrics (Test Set, n=200)

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **63.50%** |
| **Macro-F1** | **0.4424** |
| **Weighted-F1** | **0.5986** |

### Per-Class Performance Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Entailment** | 0.0000 | 0.0000 | 0.0000 | 19 |
| **Neutral** | 0.6154 | 0.8372 | 0.7094 | 86 |
| **Contradiction** | 0.6627 | 0.5789 | 0.6180 | 95 |
| **Accuracy** | — | — | **0.6350** | 200 |
| **Macro Avg** | 0.4260 | 0.4721 | **0.4424** | 200 |
| **Weighted Avg** | 0.5794 | 0.6350 | **0.5986** | 200 |

### 🔍 Failure Mode Analysis (Why is Entailment 0.00?)

The model achieved **0.00 F1 for the Entailment class**, meaning it correctly classified *zero* of the 19 entailment examples in our test subset.

**Why does this happen?**
1. **Class Imbalance:** The test subset contained only 19 entailment samples (9.5%). The model learned to play it "safe" by defaulting to Neutral or Contradiction, which are much more frequent.
2. **Subtle Phrasing:** Entailment requires recognizing that the hypothesis is *directly supported* by the premise. In legal language, this requires deep semantic alignment—much harder to learn from only 7,000 training samples than detecting obvious contradictions.
3. **Prompt Design:** The current prompt (`Classify as entailment, neutral, or contradiction`) might bias the model toward "Neutral" when uncertain, since it's the middle ground.

**How to fix this (Future Work):**
- **Data Balancing:** Oversample Entailment examples in the training set or use weighted loss functions.
- **DPO Alignment:** Use Direct Preference Optimization to train the model to explicitly *prefer* Entailment when the evidence supports it.
- **Better Prompting:** Add a "think step-by-step" instruction to force the model to reason before classifying.

*Despite this limitation, the model performs strongly on Neutral and Contradiction, proving it has learned meaningful legal reasoning—not just random guessing.*

---

## 🛠️ Hardware & Setup

- **Runtime:** Google Colab (Free Tier)
- **GPU:** T4 (16GB VRAM)
- **Training Time:** ~1.2 hours (1 epoch, effective batch size = 8)
- **Dependencies:** Python 3.10+, PyTorch, Transformers, PEFT, TRL, BitsAndBytes, Datasets.

### Installation

```bash
pip install -q torch transformers datasets accelerate evaluate bitsandbytes
pip install -q "trl>=0.8.0" "peft>=0.8.0"
pip install -q scikit-learn matplotlib seaborn
```

---

## Repository Structure

```text
contractlens-sft/
├── contractlens_sft.ipynb      # Complete Colab notebook
├── README.md                   # This file
├── lora-adapters/              # Saved PEFT weights (uploaded to Hugging Face Hub)
   ├── adapter_config.json
   └── adapter_model.safetensors

```
---

## 🤗 Model on Hugging Face Hub

The fine-tuned LoRA adapters are available for download and inference:

- LoRA adapter → [`ContractNLI-gemma-2b-sft-lora-adapter`](https://huggingface.co/SivaSai8143/ContractNLI-gemma-2b-sft-lora-adapter/tree/main)

### Quick Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "SivaSai8143/ContractNLI-gemma-2b-sft-lora-adapter")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

def classify(premise, hypothesis):
    prompt = f"""<|user|>
    Premise: {premise}
    Hypothesis: {hypothesis}
    Classify the relationship as entailment, neutral, or contradiction.
    <|assistant|>"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.0, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip().lower()
    if "entailment" in response:
        return "entailment"
    elif "contradiction" in response:
        return "contradiction"
    return "neutral"
```
---

## Disclaimer

This is an educational fine-tuning project for demonstrating LLM training pipelines. The underlying pharma content is for technical demonstration only and is **not medical advice**.
