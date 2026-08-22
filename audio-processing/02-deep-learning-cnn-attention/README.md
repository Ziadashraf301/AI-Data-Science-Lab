# Stage 2: Deep Learning — 2D Spectrogram CNNs & Temporal Attention Heads

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Torchvision](https://img.shields.io/badge/Torchvision-ResNet18-green)](https://pytorch.org/vision)
[![Librosa](<https://img.shields.io/badge/Librosa-Mel%20Spectrograms-blue>)](https://librosa.org)

This stage treats audio representations as 2D time-frequency images (**Log-Mel Spectrograms**), systematically exploring:

1. **2D Convolutional Architectures & Regularization**: Combating severe overfitting on small datasets with SpecAugment and Mixup.
2. **Computer Vision Transfer Learning**: Adapting ImageNet-pretrained **ResNet18** backbones to single-channel spectrograms.
3. **Temporal & Sequential Modeling**: Combining CNN local feature extractors with **Bidirectional LSTMs**, **Attention Pooling**, and **Multi-Head Self-Attention (MHSA)**.

---

## 🧭 Submodule Notebooks & Architecture

```text
02-deep-learning-cnn-attention/
├── README.md                                                  # This documentation
├── 02_cnn_mel_spectrograms.ipynb                              # 5 CNN approaches, SpecAugment, Mixup, ResNet18
└── 03_cnn_attention_bilstm.ipynb                              # Shared CNN trunk + BiLSTM / Attention / MHSA heads
```

---

## 🖼️ Spectrogram & Signal Processing Pipeline

- **Audio Length**: Fixed 4.0 seconds ($88,200$ samples at $22,050\text{ Hz}$).
- **Spectrogram Dimensions**: `(128, 173)` ($128$ Mel frequency bins, $173$ temporal frames, $N_{\text{fft}}=2048$, $\text{hop}=512$).
- **Data Augmentations**:
  - **SpecAugment**: Random time and frequency band masking.
  - **Mixup**: Convex linear interpolation of input spectrograms and one-hot target vectors ($\alpha=0.2$).

---

## 📊 Benchmark Results

### Experiment Set A: 2D CNNs & Transfer Learning (`02_cnn_mel_spectrograms.ipynb`)

| # | Architecture & Regularization                                | Val Accuracy     | Test Accuracy       | Overfitting Gap  |
| - | ------------------------------------------------------------ | ---------------- | ------------------- | ---------------- |
| 1 | Baseline 2D CNN (No Regularization)                          | 54.17%           | **53.24%**    | 46.76%           |
| 2 | Regularized 2D CNN (+ BatchNorm, Dropout, Scheduler)         | 62.96%           | **60.65%**    | 30.12%           |
| 3 | Regularized 2D CNN +**SpecAugment & Mixup**            | 68.06%           | **67.59%**    | 20.45%           |
| 4 | Transfer Learning (Pretrained ImageNet ResNet18)             | 71.30%           | **68.52%**    | 24.15%           |
| 5 | **Transfer Learning + SpecAugment + Mixup (ResNet18)** | **86.57%** | **87.04%** ⭐ | **12.96%** |

---

### Experiment Set B: Temporal & Attention Heads on Shared Trunk (`03_cnn_attention_bilstm.ipynb`)

All models share a common convolutional trunk `CNNFeatureExtractor` that shrinks the frequency axis into a temporal sequence `(Batch, 43, 32)`:

```text
Log-Mel Spectrogram (B, 1, 128, 173)
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ Shared CNNFeatureExtractor                             │
│ 5 Conv2D Blocks -> Average over Frequency Axis         │
│ Output: Sequence of shape (B, 43 time frames, 32 dim)  │
└────────────────────────┬───────────────────────────────┘
                         │
        ┌────────────────┼────────────────┬────────────────┐
        ▼                ▼                ▼                ▼
   [ BiLSTM ]    [ BiLSTM + Attn ]    [ MHSA ]     [ BiLSTM + MHSA ]
        │                │                │                │
     75.46%           76.85%           76.85%           77.31%
```

| Experiment  | Head Architecture                               | Test Accuracy    | Core Advantage                              |
| ----------- | ----------------------------------------------- | ---------------- | ------------------------------------------- |
| **1** | CNN +**BiLSTM**                           | **75.46%** | Models temporal order and prosodic cadence  |
| **2** | CNN +**BiLSTM + Attention Pooling**       | **76.85%** | Learns which emotional frames matter most   |
| **3** | CNN +**Multi-Head Self-Attention (MHSA)** | **76.85%** | Direct frame-to-frame global interactions   |
| **4** | CNN +**BiLSTM + Multi-Head Attention**    | **77.31%** | Fuses recurrence with global self-attention |
