# Stage 1: Classical Machine Learning on Handcrafted Audio Descriptors

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Librosa](<https://img.shields.io/badge/Librosa-Feature%20Extraction-green>)](https://librosa.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Classification-F7931E?logo=scikitlearn)](https://scikit-learn.org)

An empirical benchmark evaluating classical machine learning models and statistical classifiers on handcrafted spectral and temporal acoustic descriptors extracted from the **RAVDESS dataset** (1,440 audio clips, 8 emotion classes).

---

## 🎧 Feature Engineering Pipeline

Audio clips are sampled at **22,050 Hz** and processed with `librosa` to extract high-dimensional acoustic feature vectors:

```text
Raw Audio Waveform (22,050 Hz)
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ 1. Time-Frequency Spectral Feature Extraction          │
│    - MFCCs (Mel-Frequency Cepstral Coefficients, 40)   │
│    - Chroma STFT (12 pitch classes)                    │
│    - Mel Spectrogram (128 mel frequency bins)          │
│    - Spectral Contrast (7 frequency bands)             │
│    - Tonnetz (6 tonal centroid features)               │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Temporal Aggregation & Global Pooling               │
│    - Mean & Variance pooled across time frames         │
│    - Output: 1D Feature Vector per audio clip          │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Scaled & Normalized Dataset Matrix (1440 × D)       │
│    - Train/Val/Test Split: 70 / 15 / 15                │
└────────────────────────────────────────────────────────┘
```

---

## 📊 Models Evaluated & Benchmark Results

12 distinct classical machine learning algorithms were trained and benchmarked:

| Model Architecture                                      | Validation Accuracy | Test Accuracy    | Overfitting Gap (Train - Test) |
| ------------------------------------------------------- | ------------------- | ---------------- | ------------------------------ |
| **Multi-Layer Perceptron (MLP Neural Net)**       | **74.31%**    | **73.96%** | 26.04%                         |
| **Linear Support Vector Classifier (Linear SVM)** | 71.76%              | **71.18%** | 28.82%                         |
| **LightGBM (Gradient Boosting)**                  | 69.91%              | **70.14%** | 29.86%                         |
| **Random Forest Classifier**                      | 66.20%              | 65.74%           | 34.26%                         |
| **Logistic Regression (Softmax)**                 | 65.74%              | 64.81%           | 31.48%                         |
| **CatBoost Classifier**                           | 68.06%              | 67.13%           | 32.87%                         |
| **XGBoost Classifier**                            | 67.59%              | 66.67%           | 33.33%                         |
| **K-Nearest Neighbors (KNN)**                     | 58.80%              | 56.48%           | 24.52%                         |
| **Gaussian Naive Bayes**                          | 42.13%              | 40.74%           | 18.06%                         |

---

## 💡 Key Architectural Insights & Limitations

1. **Loss of Dynamic Temporal Context**:
   - Averaging spectral descriptors across time collapses pitch inflections, cadence changes, and energy spikes into a single static summary vector.
2. **Severe Overfitting on Small Datasets**:
   - Handcrafted feature tables have high dimensionality relative to the sample size (1,440 samples), leading classical tree ensembles and MLPs to memorize the training data (train-test gap $>25\%$).
3. **Motivation for Stage 2**:
   - Explicit 2D time-frequency spectrogram representations with convolutional filters (CNNs) are required to capture local spectro-temporal patterns without destructive temporal averaging.
