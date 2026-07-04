# 🚀 LLM Fine-Tuning Playbook
**End-to-End Fine-Tuning Pipelines for Domain-Specific LLMs**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sivasaiyadav8143/llm-finetuning-playbook/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/)
[![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&color=525252)](https://colab.research.google.com/)

---

## 📌 **About This Repository**
This repository is a **comprehensive playbook** for fine-tuning **Large Language Models (LLMs)** for **domain-specific tasks** using **parameter-efficient methods** like **LoRA, QLoRA, SFT, and DPO**. It includes **two end-to-end projects** demonstrating how to adapt LLMs for **pharma** and **legal** domains, with a focus on **reproducibility, rigor, and real-world applicability**.

---

## 🎯 **Projects**
### 1. **[Pharma-Domain LLM Fine-Tuning: 3-Stage Pipeline](https://github.com/sivasaiyadav8143/llm-finetuning-playbook/tree/main/tinyllama-pharma-3stage-finetuning)**
**Goal:** Adapt **TinyLlama-1.1B** to the **pharma domain** using a **3-stage fine-tuning pipeline**:
- **Stage 1: Non-Instruction Fine-Tuning (Continued Pretraining)**
  - **Task:** Domain adaptation via causal language modeling on raw pharma text.
  - **Data:** Paragraphs extracted from a pharma corpus (e.g., Metformin, lipid-lowering therapy, mRNA vaccines).
  - **Output:** A model specialized in pharma terminology and writing style.

- **Stage 2: Instruction Fine-Tuning (SFT)**
  - **Task:** Teach the model to follow instructions and answer pharma-related questions.
  - **Data:** Alpaca-style `instruction → response` pairs (e.g., "Explain the mechanism of action of Metformin.").
  - **Output:** A model capable of structured Q&A in the pharma domain.

- **Stage 3: Preference Tuning (DPO)**
  - **Task:** Align the model to prefer **higher-quality, factually accurate** answers over plausible but incorrect ones.
  - **Data:** `prompt → chosen vs. rejected` pairs (e.g., accurate vs. hallucinated responses).
  - **Output:** A preference-aligned model for reliable pharma reasoning.

**Key Features:**
✅ **End-to-end pipeline** from raw text to preference-tuned model.
✅ **QLoRA (4-bit quantization)** for training on **free Colab GPUs (T4)**.
✅ **LoRA adapters** for parameter-efficient fine-tuning.
✅ **Same corpus across all stages** for meaningful before/after comparisons.

**Artifacts:**
- [LoRA Adapters](https://huggingface.co/SivaSai8143) (Hugging Face Hub)
- Merged models for each stage (e.g., `pharma-tinyllama-non-instruction-merged`).

---

### 2. **[ContractLens-SFT: Legal Contract NLI](https://github.com/sivasaiyadav8143/llm-finetuning-playbook/tree/main/gemma-contractnli-sft)**
**Goal:** Fine-tune **Gemma-2B** for **Legal Natural Language Inference (NLI)** on the **ContractNLI** dataset.
- **Task:** Classify the relationship between a **contract clause (`premise`)** and a **legal statement (`hypothesis`)** as:
  - `entailment` (clause supports the statement),
  - `neutral` (no clear relationship),
  - `contradiction` (clause contradicts the statement).

**Key Features:**
✅ **Supervised Fine-Tuning (SFT)** for classification tasks.
✅ **LoRA + QLoRA** for efficient training on **free Colab GPUs (T4)**.
✅ **Rigorous evaluation** with **Accuracy, Macro-F1, and per-class metrics**.
✅ **Failure mode analysis** (e.g., handling class imbalance in `entailment`).

**Results:**
   Metric          | Score      |
 |-----------------|------------|
 | **Accuracy**    | 63.50%     |
 | **Macro-F1**    | 0.4424     |
 | **Weighted-F1** | 0.5986     |

**Artifacts:**
- [LoRA Adapter: `ContractNLI-gemma-2b-sft-lora-adapter`](https://huggingface.co/SivaSai8143/ContractNLI-gemma-2b-sft-lora-adapter)
- Full Colab notebook for reproducibility.

---

## 🔧 **Technologies & Methodologies**
 | **Component**               | **Details**                                                                                     |
 |-----------------------------|-------------------------------------------------------------------------------------------------|
 | **Models**                  | TinyLlama-1.1B, Gemma-2B                                                                       |
 | **Fine-Tuning Methods**     | LoRA, QLoRA (4-bit), SFT, DPO                                                                   |
 | **Frameworks**              | `transformers`, `peft`, `trl`, `datasets`, `accelerate`, `bitsandbytes`                       |
 | **Hardware**                | Google Colab (Free GPU: T4)                                                                    |
 | **Evaluation Metrics**     | Accuracy, Macro-F1, Weighted-F1, Confusion Matrix, BERTScore, Qualitative Review            |
 | **Datasets**                | Pharma Corpus (custom), [ContractNLI](https://huggingface.co/datasets/kiddothe2b/contract-nli) |

---


## 📂 **Repository Structure**

```text
llm-finetuning-playbook/
├── tinyllama-pharma-3stage-finetuning/  # 3-stage pharma fine-tuning pipeline
│   ├── data/                              # Raw and processed pharma data
│   │   ├── Metformin-Lipid-Therapy-Knowledge.pdf
│   │   ├── pharma_paragraph_process.jsonl
│   │   ├── pharma_instruction_dataset.jsonl
│   │   └── pharma_preference_dataset.jsonl
│   ├── huggingface/                       # Hugging Face + PEFT + TRL notebooks
│   │   └── 1_Non_Instruction_Causal_LM_Fine_Tuning_or_Domain_Adaptive_Continued_Pretraining.ipynb
│   └── README.md                          # Pharma project details
│
├── gemma-contractnli-sft/                # Legal NLI fine-tuning
│   ├── lora-adapters/                     # Saved LoRA weights
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   ├── gemma_contract_nli_lora_sft.ipynb   # Colab notebook
│   ├── unsloth_gemma_contract_nli_lora_sft.ipynb  # Unsloth Complete Colab notebook
│   └── README.md                          # Legal project details
│
├── .gitignore
└── README.md                             # This file

```

---

## 🎓 **Key Learnings & Skills Demonstrated**
### **Technical Skills**
- **Fine-Tuning LLMs:** End-to-end pipelines for **domain adaptation, instruction-following, and preference alignment**.
- **Parameter-Efficient Tuning:** LoRA, QLoRA, and adapter merging.
- **Evaluation Rigor:** Metrics for classification (F1, Accuracy) and generation (BERTScore, qualitative review).
- **Hardware Optimization:** Training on **free Colab GPUs** with 4-bit quantization.

### **Domain Expertise**
- **Pharma:** Fine-tuning for **medical/pharma terminology** and Q&A.
- **Legal:** Fine-tuning for **contract understanding** and Natural Language Inference (NLI).

### **Engineering Best Practices**
- **Reproducibility:** Full Colab notebooks with step-by-step instructions.
- **Modularity:** Separate stages for **continued pretraining, SFT, and DPO**.
- **Documentation:** Clear READMEs, comments, and artifact saving.
---