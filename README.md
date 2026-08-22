# 🧪 AI & Data Science Lab

[![Python](<https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python>)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![NumPy](<https://img.shields.io/badge/NumPy-from%20scratch-013243?logo=numpy>)](https://numpy.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Validation-F7931E?logo=scikitlearn)](https://scikit-learn.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A rigorous research and implementation laboratory exploring **first-principles mathematical algorithms**, **statistical machine learning**, **unsupervised manifold learning**, and **modern Deep Learning / Foundation Speech AI architectures**.

---

## 🧭 Repository Architecture

```text
AI-Data-Science-Lab/
│
├── pca/                                        # Principal Component Analysis
│   ├── README.md                               # Detailed mathematical & empirical study
│   ├── PCA.ipynb                               # SVD & Eigenvalue PCA from scratch + validation
│   └── breast_cancer_dataset.csv               # Wisconsin Breast Cancer Dataset (569 × 30)
│
├── tsne/                                       # t-Distributed Stochastic Neighbor Embedding
│   ├── README.md                               # Theory, KL divergence gradients & perplexity analysis
│   ├── tsne_from_scratch.ipynb                 # t-SNE implemented from scratch in pure NumPy
│   └── tsne_comparison.png                     # Visual projection comparison (Custom vs. Sklearn)
│
├── probabilistic_classifiers/                  # Probabilistic & Discriminative Classifiers
│   ├── README.md                               # Derivations, decision boundaries & small-sample analysis
│   └── probabilistic_classifiers.ipynb         # LDA, QDA, Naive Bayes & Logistic Reg from scratch
│
└── audio-processing/                           # Speech AI & Speech Emotion Recognition (SER) Track
    ├── README.md                               # Master track overview & unified leaderboard
    ├── 01-classical-audio-ml/                  # Stage 1: Classical ML & Handcrafted DSP Features
    │   ├── README.md                           # Guide on DSP extraction (MFCC, Chroma, Tonnetz) & baseline ML
    │   └── 01_classical_ml_features.ipynb      # 12 classical ML models benchmarked on pooled 1D features
    ├── 02-deep-learning-cnn-attention/         # Stage 2: 2D Spectrogram CNNs & Temporal Attention Heads
    │   ├── README.md                           # Guide on Mel-Spectrograms, SpecAugment, Mixup, BiLSTM, MHSA
    │   ├── 02_cnn_mel_spectrograms.ipynb       # 5 CNN approaches, SpecAugment, Mixup, ResNet18
    │   └── 03_cnn_attention_bilstm.ipynb      # Shared CNN trunk + BiLSTM / Attention / MHSA heads
    ├── 03-foundation-ssl-transformers/         # Stage 3: Modern SSL Speech Transformers & Foundation Models
    │   ├── README.md                           # Guide on SSL models (Wav2Vec2, HuBERT, WavLM, Whisper)
    │   ├── 04_ssl_speech_transformers.ipynb    # Probing & Fine-tuning Wav2Vec2, HuBERT, WavLM, Whisper
    │   └── wav2vec2-deep-dive/                 # Complete PyTorch wav2vec 2.0 implementation & master guides
    │       ├── README.md
    │       ├── wav2vec2_implementation.ipynb
    │       ├── wav2vec2_master_guide.md
    │       ├── wav2vec2_summary.md
    │       └── wav2vec 2.0.pdf
    └── docs/                                   # Research Roadmaps & Strategy
        ├── speech_ai_milestones.md             # Speech AI Roadmap (2015–2026)
        └── speech_emotion_recognition_plan.md  # SER Evolution & Architecture Design Plan
```

---

## 🔬 Research Pillars & Modules

### 1. 🧬 Dimensionality Reduction & Manifold Learning

| Module                                                                                              | Core Algorithm                         | Mathematical Foundation                                                                                                                     | Key Result / Dataset                                                                                                       |
| --------------------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| [**`pca/`**](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/pca/README.md)   | **Principal Component Analysis** | SVD ($\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$) & Eigendecomposition ($\mathbf{\Sigma}_{cov}\mathbf{v} = \lambda\mathbf{v}$) | **90% feature reduction** (30 $\to$ 3) with $<1\%$ accuracy loss and identical recall on malignant breast cancer |
| [**`tsne/`**](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/tsne/README.md) | **t-Distributed SNE**            | High-D Gaussian perplexity binary search, Student-t kernel, KL divergence gradient descent                                                  | Custom NumPy implementation achieves identical 2D cluster separation to`sklearn.manifold.TSNE`                           |

---

### 2. 📊 Statistical & Probabilistic Machine Learning

| Module                                                                                                                                        | Models Implemented                                            | Mathematical Focus                                                                                         | Evaluation Regimes                                                                                       |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [**`probabilistic_classifiers/`**](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/probabilistic_classifiers/README.md) | **LDA, QDA, Gaussian Naive Bayes, Logistic Regression** | Shared vs. class-specific covariance matrices, conditional independence assumptions, log-odds optimization | $p=1$ boundary visualization, $p>1$ high-D cross validation, and small-sample breakdown ($n < 30$) |

---

### 3. 🎙️ Speech AI & Audio Foundation Models

| Stage & Submodule                                                                                                                                                             | Paradigm                               | Key Architectures & Techniques                                                                      | Performance (RAVDESS)                             |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| [**Stage 1: Classical Audio ML**](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/audio-processing/01-classical-audio-ml/README.md)                       | Handcrafted Spectral Descriptors       | MFCCs, Chroma, Tonnetz, Spectral Contrast, Tree Ensembles, MLP                                      | 73.96% Test Acc                                   |
| [**Stage 2: Spectrogram CNNs & Attention**](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/audio-processing/02-deep-learning-cnn-attention/README.md)    | Spectrogram Vision & Sequence Modeling | Log-Mel Spectrograms, SpecAugment, Mixup, ResNet18, BiLSTM, Attention Pooling, Multi-Head Attention | 87.04% Test Acc (ResNet18) / 77.31% (BiLSTM+MHSA) |
| [**Stage 3: Foundation SSL Transformers**](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/audio-processing/03-foundation-ssl-transformers/README.md)     | Self-Supervised Raw Waveform Models    | Pre-trained Wav2Vec 2.0, HuBERT, WavLM, Whisper (Probing & End-to-End Fine-Tuning)                  | **88.89%+ Test Acc** ⭐                     |
| [**wav2vec 2.0 Deep Dive**](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/audio-processing/03-foundation-ssl-transformers/wav2vec2-deep-dive/README.md) | Foundation Speech Architecture         | 7-layer 1D Temporal CNN, Product Quantization (Gumbel-Softmax), Masked Span Transformer             | Complete PyTorch implementation from scratch      |

---

## 📊 Benchmark Summary Leaderboard

### Speech Emotion Recognition (RAVDESS 8-Class Benchmark)

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ Model Architecture                                     │ Test Accuracy   │
├──────────────────────────────────────────────────────────────────────────┤
│ Classical Baseline (Handcrafted MFCC + MLP)            │ 73.96%          │
│ Baseline 2D CNN (No Regularization)                    │ 53.24%          │
│ Regularized 2D CNN (+ BatchNorm, Dropout, Scheduler)   │ 60.65%          │
│ Regularized 2D CNN + Augmentation (SpecAugment, Mixup) │ 67.59%          │
│ Transfer Learning (Pretrained ImageNet ResNet18)       │ 68.52%          │
│ Transfer Learning + Augmentation (ResNet18 + Mixup)    │ 87.04%          │
│ Wav2Vec 2.0 (Frozen Encoder / Linear Probing)          │ 66.20%          │
│ HuBERT (Frozen Encoder / Linear Probing)               │ 70.83%          │
│ Wav2Vec 2.0 (Full Fine-Tuning End-to-End)              │ 88.89% ⭐       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack & Dependencies

- **Core Languages**: Python 3.10+
- **Scientific Computing**: NumPy, SciPy, Pandas, Statsmodels
- **Machine Learning**: Scikit-Learn
- **Deep Learning**: PyTorch, Torchvision, Torchaudio
- **Audio Processing**: Librosa, SoundFile, Hugging Face Transformers & Datasets
- **Visualization**: Matplotlib, Seaborn

---

## 🚀 Quickstart

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Ziadashraf301/AI-Data-Science-Lab.git
cd AI-Data-Science-Lab
```

### 2. Install Dependencies

```bash
pip install numpy scipy pandas scikit-learn matplotlib seaborn statsmodels
pip install torch torchvision torchaudio transformers datasets librosa soundfile
```

### 3. Launch Interactive Notebooks

```bash
jupyter lab
```

---

## 👤 Author

**Ziad Ashraf**
*AI & Machine Learning Engineer/ Data Scientist*
GitHub: [@Ziadashraf301](https://github.com/Ziadashraf301)
