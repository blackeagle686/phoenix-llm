# Phoenix-LLM 🚀

A from-scratch implementation of a high-performance, decoder-only Transformer language model, built on top of the custom **Phoenix Engine** framework.

## 📖 Project Overview

Phoenix-LLM is designed to be a modular, scalable, and highly optimized LLM platform. Following a phase-by-phase build plan, we are constructing every component of the stack—from the data pipeline and tokenizer to the distributed training loop and streaming inference API.

### Current Progress: **Phase 3 (Model Architecture)**
We have completed the foundational architecture, implementing a modern transformer stack optimized for the Phoenix Engine backend.

## ✨ Key Features

- **Consolidated GQA/MHA:** An optimized Grouped Query Attention module that generalizes Multi-Head Attention for better inference efficiency.
- **SwiGLU FFN:** Implementation of the state-of-the-art Gated Linear Unit activation structure.
- **Pre-LN Architecture:** Improved training stability using Pre-Normalization with RMSNorm support.
- **Weight Tying:** Shared weights between token embeddings and the LM head to reduce parameter count and improve convergence.
- **Modular Design:** Independent components for attention, FFNs, normalization, and positional encodings.

## 📁 Project Structure

```text
├── components/           # Modular model components
│   ├── attention/        # GQA, MHA, and Flash Attention
│   ├── ffn/              # SwiGLU and MoE implementations
│   ├── normalization/    # RMSNorm and LayerNorm
│   └── positional/       # RoPE and Learned Embeddings
├── core/                 # Core configuration and utilities
├── models/               # Full model architectures (GPT, etc.)
├── tests/                # Architecture and unit tests
├── data/                 # Data pipeline and ingestion (Phase 1)
├── training/             # Training loops and optimizers (Phase 4)
└── plan.md               # Master build specification
```

## 🛠️ Tech Stack

- **Engine:** [Phoenix Engine](https://github.com/blackeagle686/phoenix-engine) (Custom C++ Backend)
- **Language:** Python 3.11+
- **Architecture:** Decoder-only Transformer
- **Normalization:** RMSNorm
- **Positional Encodings:** RoPE (Rotary Position Embeddings)

##  Getting Started

### 1. Prerequisites
Ensure you have the `phoenix-engine` installed in your environment.

### 2. Configuration
Hyperparameters are managed via `core/config.py`. You can adjust model size, heads, and layers there.

### 3. Verify Architecture
Run the architecture verification script to ensure the model initializes correctly:
```bash
python tests/test_architecture.py
```

## 🗺️ Roadmap

- [x] **Phase 3:** Model Architecture (Base)
- [ ] **Phase 1:** Data Pipeline (Ingestion, Cleaning, Deduplication)
- [ ] **Phase 2:** Tokenizer Training (BPE)
- [ ] **Phase 4:** Training Loop (AdamW, Cosine Decay, Mixed Precision)
- [ ] **Phase 6:** Inference & Serving (KV Cache, SSE Streaming)

---
*Built with speed and precision on the Phoenix Engine.*
