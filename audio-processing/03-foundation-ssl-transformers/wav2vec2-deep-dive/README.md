# wav2vec 2.0 — Deep Architectural Reference & Implementation

[![PyTorch](<https://img.shields.io/badge/PyTorch-Pure%20Implementation-EE4C2C?logo=pytorch>)](https://pytorch.org)
[![Paper](<https://img.shields.io/badge/NeurIPS%202020-Paper%20PDF-red>)](<wav2vec%202.0.pdf>)

> **Reference Paper:** *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*
> **Authors:** Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli (Meta AI, NeurIPS 2020)

---

## 🏗️ Architecture & Dataflow

```text
[ Raw Audio Waveform: X ]  (16 kHz continuous acoustic signal)
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ 1. Temporal Convolutional Feature Encoder f: X ──► Z    │
│    - 7 blocks of 1D convolutions (512 channels)        │
│    - Strides: (5, 2, 2, 2, 2, 2, 2) ➔ Total stride = 320  │
│    - Output: Latent frames z_t (25ms window, 20ms stride)│
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│ Masking Span Subsampling │  │ Quantization Module      │
│ - Consecutive time spans │  │ - G=2 codebooks          │
│ - ~49% of frames masked  │  │ - V=320 entries each     │
│ - Replaced with mask emb │  │ - Gumbel-Softmax lookup  │
└────────────┬─────────────┘  └──────────┬───────────────┘
             │                           │
             ▼                           ▼
┌──────────────────────────┐     [ Discrete Quantized ]
│ Transformer Context Net  │     [ Representations: q ]
│ - Multi-Head Attention   │             │
│ - Output: c_t            │             │
└────────────┬─────────────┘             │
             │                           │
             └─────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐
              │ Contrastive Loss L_m     │
              │ + Diversity Loss L_d     │
              └──────────────────────────┘
```

---

## 📁 Submodule Contents

| File                                                                        | Description                                                                                                                                                                          |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [**`wav2vec2_implementation.ipynb`**](wav2vec2_implementation.ipynb) | Complete, runnable**PyTorch implementation from first principles** (CNN encoder, Product Quantizer, Span Masker, Transformer, Contrastive Loss).                               |
| [**`wav2vec2_master_guide.md`**](wav2vec2_master_guide.md)           | **375-line master reference** covering full tensor dimensions, Gumbel-Softmax temperature annealing math, contrastive loss derivations, and downstream fine-tuning procedures. |
| [**`wav2vec2_summary.md`**](wav2vec2_summary.md)                     | High-level executive study guide and cheat sheet for rapid conceptual review.                                                                                                        |
| [**`wav2vec 2.0.pdf`**](<wav2vec%202.0.pdf>)                         | Original NeurIPS 2020 paper.                                                                                                                                                         |
