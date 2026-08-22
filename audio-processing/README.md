# 🎙️ Speech AI & Speech Emotion Recognition (SER) Track

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface)](https://huggingface.co)
[![Librosa](https://img.shields.io/badge/Librosa-Audio%20DSP-green)](https://librosa.org)

A comprehensive, evolutionary research benchmark on **Speech Emotion Recognition (SER)** using the **RAVDESS dataset** (1,440 audio clips, 8 emotions, 24 actors), organized into **3 distinct paradigm stages**: from classical acoustic DSP features to 2D Spectrogram Vision models, Temporal Attention heads, and modern **Self-Supervised Learning (SSL) Speech Transformers**.

---

## 🧭 Pipeline Architecture & Stages

```text
audio-processing/
│
├── 01-classical-audio-ml/                             # Stage 1: Classical ML & Handcrafted DSP Features
│   ├── README.md                                      # Guide on DSP extraction (MFCCs, Chroma, Tonnetz) & baseline ML
│   └── 01_classical_ml_features.ipynb                 # 12 classical ML models benchmarked on pooled 1D features
│
├── 02-deep-learning-cnn-attention/                    # Stage 2: 2D Spectrogram CNNs & Temporal Attention Heads
│   ├── README.md                                      # Guide on Mel-Spectrograms, SpecAugment, Mixup, BiLSTM, MHSA
│   ├── 02_cnn_mel_spectrograms.ipynb                  # 5 CNN approaches, SpecAugment, Mixup, ResNet18 (87.04%)
│   └── 03_cnn_attention_bilstm.ipynb                 # Shared CNN trunk + BiLSTM / Attention / MHSA heads
│
├── 03-foundation-ssl-transformers/                    # Stage 3: Modern SSL Speech Transformers & Foundation Models
│   ├── README.md                                      # Guide on SSL models (Wav2Vec2, HuBERT, WavLM, Whisper)
│   ├── 04_ssl_speech_transformers.ipynb               # Probing & Fine-tuning Wav2Vec2, HuBERT, WavLM, Whisper (88.89%)
│   └── wav2vec2-deep-dive/                            # Complete PyTorch wav2vec 2.0 implementation & master guides
│       ├── README.md
│       ├── wav2vec2_implementation.ipynb
│       ├── wav2vec2_master_guide.md
│       ├── wav2vec2_summary.md
│       └── wav2vec 2.0.pdf
│
└── docs/                                              # Research Roadmaps & Strategy
    ├── speech_ai_milestones.md                        # Speech AI Roadmap (2015–2026)
    └── speech_emotion_recognition_plan.md             # SER Evolution & Architecture Design Plan
```

---

## 🏆 Unified Benchmark Leaderboard (RAVDESS 8-Class)

| Stage | Paradigm | Model Architecture | Validation Acc | Test Acc | Overfitting Gap |
|---|---|---|---|---|---|
| **01** | Classical ML | Handcrafted Spectral Descriptors + **MLP** | — | **73.96%** | 26.04% |
| **01** | Classical ML | Handcrafted Descriptors + **Linear SVM** | — | **71.18%** | 28.82% |
| **01** | Classical ML | Handcrafted Descriptors + **LightGBM** | — | **70.14%** | 29.86% |
| **02** | 2D CNN | Baseline 2D CNN (No Regularization) | 54.17% | **53.24%** | 46.76% |
| **02** | 2D CNN | Regularized CNN (+ BatchNorm, Dropout, Scheduler) | 62.96% | **60.65%** | 30.12% |
| **02** | 2D CNN | Regularized CNN + **SpecAugment & Mixup** | 68.06% | **67.59%** | 20.45% |
| **02** | Transfer Learning | Pre-trained ImageNet **ResNet18** | 71.30% | **68.52%** | 24.15% |
| **02** | Transfer Learning | **ResNet18 + SpecAugment + Mixup** | **86.57%** | **87.04%** | **12.96%** |
| **02** | Temporal & Attention | CNN Trunk + **BiLSTM & Attention Pooling** | 76.85% | **75.46%** | 15.20% |
| **02** | Temporal & Attention | CNN Trunk + **Multi-Head Self-Attention (MHSA)** | 77.78% | **76.85%** | 14.80% |
| **03** | SSL Foundation Model | **Wav2Vec 2.0** (Frozen Encoder / Linear Probing) | 64.81% | **66.20%** | 9.69% |
| **03** | SSL Foundation Model | **HuBERT** (Frozen Encoder / Linear Probing) | 66.20% | **70.83%** | 4.96% |
| **03** | SSL Foundation Model | **Wav2Vec 2.0 (Full Fine-Tuning End-to-End)** | **89.35%** | **88.89%** ⭐ | **10.71%** |

---

## 🔬 Key Takeaways Across the 3 Stages

1. **Stage 1 (Feature Extraction)**: Demonstrated that while spectral features carry emotional markers, time-averaged pooling irreversibly destroys temporal prosody and suffers from severe overfitting.
2. **Stage 2 (Spectrogram Vision & Attention)**: 2D representations with aggressive regularization (SpecAugment, Mixup) and computer vision transfer learning (ResNet18) drastically improve generalization to **87.04%**. Temporal attention heads isolate emotionally critical syllables.
3. **Stage 3 (Foundation Speech Models)**: Training directly on raw 16kHz audio waveforms using pre-trained Self-Supervised models (Wav2Vec 2.0) bypasses visual spectrogram approximations and sets the new benchmark at **88.89% test accuracy**.

---

## 🚀 Quick Navigation

- [**Stage 1: Classical Audio ML**](01-classical-audio-ml/README.md)
- [**Stage 2: Deep Learning — Spectrogram CNNs & Attention**](02-deep-learning-cnn-attention/README.md)
- [**Stage 3: Foundation SSL Speech Transformers**](03-foundation-ssl-transformers/README.md)
- [**wav2vec 2.0 Deep Dive Reference**](03-foundation-ssl-transformers/wav2vec2-deep-dive/README.md)
- [**Speech AI Milestones Roadmap (2015–2026)**](docs/speech_ai_milestones.md)
- [**SER Notebook Evolution Plan**](docs/speech_emotion_recognition_plan.md)
