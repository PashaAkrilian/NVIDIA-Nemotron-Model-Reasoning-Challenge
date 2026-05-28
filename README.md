# NVIDIA Nemotron Model Reasoning Challenge





A collaborative deep learning and LLM fine-tuning project developed for the **NVIDIA Nemotron Model Reasoning Challenge**.

This project focuses on improving the reasoning capability of the **Nemotron-3-Nano-30B** model using:

- Chain-of-Thought (CoT) reasoning
- LoRA fine-tuning
- supervised instruction tuning (SFT)
- structured reasoning datasets
- optimized Kaggle GPU training pipelines

The notebook implements a complete reasoning enhancement workflow including:

- dataset curation
- tokenizer initialization
- LoRA configuration
- model training
- output validation
- Kaggle-ready packaging

---

# Team Information

This project was collaboratively developed by:

- **Dimas Pasha Akrilian**
- **M Irfan Al Hakim**

from:

**Universitas Negeri Semarang**

---

# Project Overview

Large Language Models (LLMs) often struggle with:

- multi-step reasoning
- symbolic logic
- arithmetic consistency
- structured problem decomposition
- hallucinated intermediate reasoning

To address these limitations, this project fine-tunes NVIDIA’s:

```text
Nemotron-3-Nano-30B
```

using:

```text
Parameter-Efficient Fine-Tuning (PEFT)
```

through:

```text
LoRA (Low-Rank Adaptation)
```

combined with:

```text
Chain-of-Thought (CoT) reasoning supervision
```

The training pipeline is optimized for Kaggle GPU environments and supports constrained-resource experimentation.

---

# Repository Structure

```bash
NVIDIA-Nemotron-Model-Reasoning/
│
├── nvidia-nemotron-model-reasoning.ipynb
├── Model Reasoning Dataset/
│   ├── train.csv
│   └── test.csv
├── LICENSE
└── README.md
```

---

# Workflow

```text
Load reasoning dataset
        ↓
Data curation & CoT formatting
        ↓
Tokenizer initialization
        ↓
Nemotron model loading
        ↓
LoRA configuration
        ↓
Supervised fine-tuning (SFT)
        ↓
Output validation
        ↓
Model archiving
        ↓
Kaggle submission packaging
```

---

# Methodology

# 1. Data Curation & CoT Formatting

The project uses curated reasoning datasets stored in:

```text
train.csv
```

and:

```text
test.csv
```

The training data is transformed into:

```text
Instruction → Reasoning → Final Answer
```

format to improve multi-step reasoning behavior.

This enables the model to learn:

- logical decomposition
- structured inference
- intermediate reasoning traces
- consistent answer generation

---

# 2. Tokenizer Initialization

The notebook dynamically resolves the local path of the Nemotron tokenizer using:

```python
KaggleHub
```

This avoids hardcoded paths and improves portability across Kaggle runtime environments.

Special tokens are explicitly configured to maintain instruction-tuning consistency.

---

# 3. LoRA Fine-Tuning

The project applies:

```text
Low-Rank Adaptation (LoRA)
```

for parameter-efficient training.

This approach significantly reduces:

- VRAM consumption
- trainable parameters
- GPU memory requirements
- training overhead

while preserving strong reasoning performance.

---

# 4. Nemotron-3-Nano-30B

Base model used:

```text
Nemotron-3-Nano-30B
```

The model is designed for:

- instruction following
- reasoning tasks
- generative AI workflows
- conversational intelligence

The notebook includes:

- CUDA runtime patching
- Triton integration
- optimized transformer loading
- GPU-aware memory handling

for stable large-model training.

---

# 5. Supervised Fine-Tuning (SFT)

The fine-tuning pipeline uses:

```text
TRL + Transformers
```

for supervised instruction training.

The model learns to:

- generate structured reasoning chains
- solve multi-step prompts
- improve logical consistency
- reduce hallucinated outputs

---

# 6. Output Validation

The notebook validates generated responses using:

- regex-based formatting checks
- answer structure validation
- reasoning-output consistency checking

This ensures outputs follow competition formatting requirements.

---

# Example Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)
```

### LoRA Training

```python
trainer.train()
```

### Inference Example

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=256
)
```

---

# Technologies Used

- Python
- PyTorch
- Transformers
- TRL
- PEFT
- LoRA
- datasets
- pandas
- Triton
- CUDA
- KaggleHub

---

# Key Features

- Chain-of-Thought fine-tuning
- Parameter-efficient training
- Kaggle-compatible workflow
- GPU-aware optimization
- Structured reasoning supervision
- Nemotron model integration
- Automatic packaging pipeline

---

# Use Cases

This project is highly relevant for:

- LLM reasoning research
- instruction tuning
- generative AI experimentation
- mathematical reasoning systems
- logical inference modeling
- Kaggle LLM competitions
- educational AI systems

---

# License

This project is licensed under the **MIT License**.

You are free to use, modify, distribute, and adapt this work with proper attribution.

---

# MIT License

Copyright (c) 2026 Dimas Pasha Akrilian & M Irfan Al Hakim

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

# Authors

## Dimas Pasha Akrilian

Universitas Negeri Semarang

## M Irfan Al Hakim

Universitas Negeri Semarang

This repository is part of our collaborative research and experimentation in:

- Large Language Models (LLMs)
- reasoning enhancement
- parameter-efficient fine-tuning
- Chain-of-Thought learning
- generative AI systems
- competition-oriented AI development

