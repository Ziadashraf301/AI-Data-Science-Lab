# Stage 3: Modern Speech Transformers & Self-Supervised Learning (SSL)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface)](https://huggingface.co)
[![State-of-the-Art](<https://img.shields.io/badge/SOTA-88.89%25%20Accuracy-success>)](https://arxiv.org/abs/2006.11477)

Stage 3 represents the modern State-of-the-Art paradigm shift in audio AI: bypassing handcrafted spectrograms entirely and feeding **16kHz raw audio waveforms** into foundation speech representation models pre-trained on thousands of hours of unlabeled speech.

---

## 🧭 Submodule Structure

```text
03-foundation-ssl-transformers/
├── README.md                                                  # This documentation
├── 04_ssl_speech_transformers.ipynb                          # Benchmark: Wav2Vec 2.0, HuBERT, WavLM, Whisper
└── wav2vec2-deep-dive/                                        # Complete from-scratch implementation & study
    ├── README.md                                              # Deep dive guide & architecture dataflow
    ├── wav2vec2_implementation.ipynb                          # Pure PyTorch wav2vec 2.0 implementation
    ├── wav2vec2_master_guide.md                               # 375-line master reference & tensor dimensions
    ├── wav2vec2_summary.md                                    # Executive cheat sheet
    └── wav2vec 2.0.pdf                                        # Original NeurIPS 2020 paper
```

---

## 🔬 SSL Models Evaluated & Pre-training Paradigms

| Model                 | Architecture                       | Pre-training Paradigm                                | Primary Strength                    |
| --------------------- | ---------------------------------- | ---------------------------------------------------- | ----------------------------------- |
| **Wav2Vec 2.0** | 1D Temporal CNN + Transformer      | Masked Contrastive Coding over Codebook Quantization | Raw waveform representation pioneer |
| **HuBERT**      | 1D Temporal CNN + BERT Transformer | Masked Prediction of Offline K-Means Pseudo-Units    | Clean phonetic acoustic codebooks   |
| **WavLM**       | 1D CNN + Conformer/Transformer     | Masked MLM + Gated Relative Bias + Speech Denoising  | SOTA for prosody & speaker traits   |
| **Whisper**     | Encoder-Decoder Transformer        | Supervised Multitasking on 680k hours                | Robust multilingual ASR embeddings  |

---

## 📊 Benchmark Results on RAVDESS (8-Class Emotion)

For each foundation model, two training regimes are evaluated:

1. **Frozen Encoder (Linear Probing)**: Tests the representation quality of frozen speech embeddings with only the classification head trained.
2. **Full End-to-End Fine-Tuning**: Adapts all transformer weights with a low learning rate ($2 \times 10^{-5}$).

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ Model & Training Mode                                   │ Test Accuracy     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Wav2Vec 2.0 (Frozen Probing)                            │ 66.20%            │
│ HuBERT (Frozen Probing)                                 │ 70.83%            │
│ HuBERT (End-to-End Fine-Tuned)                          │ 87.04%            │
│ Wav2Vec 2.0 (End-to-End Fine-Tuned)                     │ 88.89% ⭐ (SOTA)  │
└─────────────────────────────────────────────────────────────────────────────┘
```
