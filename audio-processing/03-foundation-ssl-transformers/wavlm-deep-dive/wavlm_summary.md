# WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing

## Paper Information

- **Title:** WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing
- **Authors:** Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Xiangzhan Yu, Furu Wei (Microsoft Research, Microsoft Azure Speech, Shanghai Jiao Tong University)
- **Year / Publication:** IEEE JSTSP 2022 / arXiv:2110.13900 (2021)
- **Code & Checkpoints:** [microsoft/unilm/wavlm](https://github.com/microsoft/unilm/tree/master/wavlm) | Hugging Face: `microsoft/wavlm-base-plus`, `microsoft/wavlm-large`

---

## 1. Executive Summary & Core Motivation

While prior self-supervised speech foundation models (**wav2vec 2.0**, **HuBERT**) established breakthroughs in Automatic Speech Recognition (ASR), their pre-training objectives primarily preserved **phonetic / lexical** information:

1. **Phonetic Bias in Prior SSL:** wav2vec 2.0 and HuBERT discard or marginalize background acoustics, speaker identity, and prosody in pursuit of clean word transcription.
2. **Poor Generalization on Non-ASR Tasks:** Models trained strictly on clean single-speaker speech underperform on full-stack speech benchmarks (e.g., **SUPERB**), specifically in:
   - **Speaker Verification (SV / ASV)**
   - **Speaker Diarization (SD)**
   - **Speech Emotion Recognition (ER)**
   - **Speech Enhancement & Speech Separation (SS)**

**WavLM** solves this by extending the masked speech unit prediction paradigm with two fundamental architectural and methodological innovations:

- **Masked Speech Denoising Pre-Training:** Artificially overlaps noise and interfering secondary speakers onto the raw waveform, while forcing the model to reconstruct the pseudo-phonetic discrete units of the **clean primary speaker**.
- **Gated Relative Positional Bias:** Injects a learned, gated relative position scalar directly into the Transformer self-attention matrix to better model short- and long-range acoustic dependencies.

---

## 2. Model Architecture & Dataflow

WavLM shares the temporal convolutional front-end with wav2vec 2.0 and HuBERT, but upgrades the context encoder and pre-training task:

```
          [ Clean Audio X ] ────────┐
                                    ▼
       [ Background Noise N ] ──► [ Overlap Simulation ] ──► [ Corrupted Audio X̃ ]
       [ Interfering Spk S ]                                        │
                                                                    ▼
                                                      ┌───────────────────────────┐
                                                      │    7-layer CNN Encoder    │ (320x downsampling)
                                                      └─────────────┬─────────────┘
                                                                    │
   [ Clean Audio X ]                                                ▼
          │                                           ┌───────────────────────────┐
          ▼                                           │    Span Masking Engine    │ (p = 0.08, l = 10)
  [ k-means Clusters ]                                └─────────────┬─────────────┘
  (Target Discrete IDs)                                             │
          │                                                         ▼
          │                                           ┌───────────────────────────┐
          │                                           │  Gated Relative Pos-Bias  │
          │                                           │    Transformer Encoder    │
          │                                           └─────────────┬─────────────┘
          │                                                         │
          ▼                                                         ▼
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │         Masked Cross-Entropy Loss on Clean Targets (alpha = 1.0)              │
  └───────────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Specifications:

1. **1D Temporal CNN Extractor:**
   - 7 layers, strides: `[5, 2, 2, 2, 2, 2, 2]` ($320\times$ downsampling $\implies 20\text{ ms}$ frames at $50\text{ Hz}$).
   - 512 channels across all conv layers.
2. **Gated Relative Position Transformer:**
   - Instead of purely static positional encodings, self-attention logits incorporate dynamic relative distance biases gated by query representations:
     $$
     \text{Attn}_{i, j} \propto \frac{q_i k_j^\top}{\sqrt{d}} + r_{i-j} \cdot \sigma(w_g^\top q_i + b_g)
     $$
3. **Model Configurations:**
   - **WavLM Base / Base+:** 12 Transformer layers, 768 hidden dimension, 8 attention heads, 94.7M parameters.
   - **WavLM Large:** 24 Transformer layers, 1024 hidden dimension, 16 attention heads, 316.6M parameters.

---

## 3. Key Methodological Innovations

### 1. Masked Speech Denoising & Speaker Separation Simulation

To equip the model with full-stack capability across both phonetic and non-phonetic tasks:

- During pre-training, an audio mixer dynamically combines:
  1. Primary clean speech signal $X$.
  2. Noise audio $N$ (from DNS Challenge & MUSAN) at random SNR levels ($-5\text{ dB}$ to $20\text{ dB}$).
  3. Secondary overlapping speech $S$ with $50\%$ probability to simulate conversational overlapping speech.
- **The Core Trick:** The Transformer inputs the noisy/overlapped audio, but the **k-means prediction targets are derived exclusively from the clean primary utterance $X$**.
- This forces the Transformer to perform implicit **speech denoising, separation, and speaker tracking** in its hidden states.

### 2. Gated Relative Positional Bias

Standard sinusoidal or convolution-based positional encodings treat all sequence positions uniformly. WavLM introduces a **gated relative position mechanism**:

- Computes relative offsets $|i - j|$ up to a maximum bucket limit.
- Gates the relative positional bias using a sigmoid function conditioned on current frame content.
- Allows attention heads to dynamically choose when to rely on local phonetic context versus global utterance prosody.

---

## 4. Benchmark Performance on SUPERB

WavLM achieved **Rank #1** on the universal [SUPERB Benchmark](https://superbbenchmark.org/) across all speech tasks:

| Model                       |  ASR (WER ↓)  | Speaker Identification (SID Acc ↑) | Automatic Speaker Verification (ASV EER ↓) | Speaker Diarization (SD DER ↓) | Emotion Recognition (ER Acc ↑) | Separation (SS PESQ / SISDR ↑) |
| :-------------------------- | :------------: | :---------------------------------: | :-----------------------------------------: | :-----------------------------: | :-----------------------------: | :-----------------------------: |
| **wav2vec 2.0 Base**  |      5.74      |                75.18                |                    6.02                    |              6.08              |              63.43              |              - / -              |
| **HuBERT Base**       |      6.42      |                81.42                |                    5.11                    |              5.88              |              64.92              |              - / -              |
| **WavLM Base+**       | **5.59** |           **84.51**           |               **4.69**               |         **4.66**         |         **65.94**         |      **2.67 / 9.53**      |
| **wav2vec 2.0 Large** |      3.75      |                86.14                |                    3.98                    |              5.90              |              65.64              |              - / -              |
| **HuBERT Large**      |      3.62      |                90.33                |                    3.82                    |              5.75              |              67.62              |              - / -              |
| **WavLM Large**       | **3.44** |           **95.23**           |               **2.61**               |         **3.27**         |         **70.62**         |     **3.01 / 10.51**     |

> **Key Takeaway:** While HuBERT and wav2vec 2.0 are comparable on standard ASR, WavLM demonstrates dramatic superiority on **Speaker Verification (ASV: 2.61% vs 3.82% EER)**, **Speaker Diarization (DER: 3.27% vs 5.75%)**, and **Emotion Recognition (70.62% vs 67.62%)**.

---

## 5. Architectural Comparison: wav2vec 2.0 vs. HuBERT vs. WavLM

| Dimension                     | wav2vec 2.0                     | HuBERT                                                                                  | WavLM                                                                  |
| :---------------------------- | :------------------------------ | :-------------------------------------------------------------------------------------- | :--------------------------------------------------------------------- |
| **Target Discovery**    | Online Gumbel-Softmax Codebook  | Offline$k$-means on MFCC/Latents                                                      | Offline$k$-means on Latents                                          |
| **Pre-training Loss**   | Contrastive InfoNCE + Diversity | Masked Cross-Entropy ($\alpha=1.0$) | Masked Denoising Cross-Entropy ($\alpha=1.0$) |                                                                        |
| **Noise Robustness**    | Sensitive to acoustic noise     | Moderate                                                                                | Native (trained with overlapping noise & speech)                       |
| **Positional Encoding** | Conv1D Depthwise                | Conv1D Depthwise                                                                        | **Gated Relative Position Bias**                                 |
| **Target Objective**    | Reconstruct masked latent frame | Reconstruct masked cluster ID                                                           | Reconstruct**clean** cluster ID from **noisy** input       |
| **Best Suited For**     | Clean ASR                       | Clean ASR & Phoneme extraction                                                          | Full-Stack Speech (ASR, Speaker ID, Diarization, Emotion, Enhancement) |
