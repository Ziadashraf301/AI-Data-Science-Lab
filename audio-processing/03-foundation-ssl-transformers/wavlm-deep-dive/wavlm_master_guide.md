# 📖 WavLM: Complete Master Technical & Architectural Reference

> **Paper Title:** *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*
> **Authors:** Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Xiangzhan Yu, Furu Wei (Microsoft Research, IEEE JSTSP 2022)
> **Official Checkpoints:** `microsoft/wavlm-base`, `microsoft/wavlm-base-plus`, `microsoft/wavlm-large`

---

# TABLE OF CONTENTS

1. [Executive Summary &amp; Foundational Motivation](#1-executive-summary--foundational-motivation)
2. [Complete System Architecture &amp; Dataflow](#2-complete-system-architecture--dataflow)
3. [Innovation 1: Gated Relative Position Bias (Math &amp; Mechanics)](#3-innovation-1-gated-relative-position-bias-math--mechanics)
4. [Innovation 2: Masked Speech Denoising &amp; Overlap Simulation](#4-innovation-2-masked-speech-denoising--overlap-simulation)
5. [Pre-Training Objective &amp; Multi-Iteration Pipeline](#5-pre-training-objective--multi-iteration-pipeline)
6. [SUPERB Benchmark Dominance &amp; Empirical Analysis](#6-superb-benchmark-dominance--empirical-analysis)
7. [Full Standalone PyTorch Implementation](#7-full-standalone-pytorch-implementation)
8. [Practical Fine-Tuning Guide &amp; Real-World Use Cases](#8-practical-fine-tuning-guide--real-world-use-cases)

---

# 1. Executive Summary & Foundational Motivation

### The Limitation of Wav2Vec 2.0 and HuBERT

Prior foundation speech models (Wav2Vec 2.0, HuBERT) were designed with a single primary downstream task in mind: **Automatic Speech Recognition (ASR)**.

- In ASR, the network's goal is **invariance to speaker characteristics, background noise, room reverberation, and pitch**.
- Consequently, HuBERT and Wav2Vec 2.0 actively discard non-linguistic acoustic signals (speaker timbre, acoustic emotion, spatial noise, and overlapping speaker streams).
- When evaluated on **Full-Stack Speech Processing** (Speaker Identification, Speaker Diarization, Emotion Recognition, Speech Separation, and Speech Enhancement), traditional SSL models showed suboptimal performance.

### The WavLM Paradigm: Full-Stack Speech Representation

WavLM was engineered to solve all speech processing tasks simultaneously:

1. **Speech Recognition (ASR / PR)**: Content-preserving representations.
2. **Speaker Verification & Diarization (SV / SD)**: Speaker-discriminative embeddings.
3. **Speech Emotion Recognition (SER)**: Prosody and emotional tone modeling.
4. **Speech Separation & Enhancement (SS / SE)**: Acoustic signal decomposition and noise immunity.

WavLM achieves this through two fundamental architectural and algorithmic breakthroughs:

1. **Gated Relative Position Bias in Self-Attention**: Captures precise temporal order and local phonological duration dynamically scaled per head.
2. **Masked Speech Denoising & Overlapping Simulation Pre-training**: Feeding masked noisy, multi-speaker audio into the network while forcing it to predict the discrete acoustic units of the **clean, original speaker**.

---

# 2. Complete System Architecture & Dataflow

```text
[ Input Audio: x ] + [ Noise: n ] + [ Overlapping Speech: s ]
                       │
                       ▼ (Dynamically mixed on-the-fly)
           [ Corrupted Audio Waveform: x̃ ] (16 kHz)
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ 1. 1D Temporal Convolutional Feature Extractor         │
│    - 7 blocks of 1D Convolutions (512 channels, GELU)  │
│    - Strides: (5, 2, 2, 2, 2, 2, 2) ➔ Total Stride 320 │
│    - Receptive field: 25ms window, 20ms stride (50 Hz) │
│    - Output: Latent Feature Sequence Z = [z_1, ..., z_T]│
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ 2. Span Masking Engine                                 │
│    - Sample starting indices with probability p = 0.065│
│    - Mask continuous span of length M = 10 frames      │
│    - ~49% of sequence replaced by learnable vector e_m │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ 3. Transformer / Conformer with Gated Rel-Pos Bias     │
│    - 12 Blocks (Base/Base+) or 24 Blocks (Large)       │
│    - Gated Relative Position Bias matrix R ∈ R^(T × T) │
│    - Attention: Softmax((Q K^T)/√d + R) · V            │
│    - Output: Contextual Embeddings C = [c_1, ..., c_T] │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ 4. MLM Cross-Entropy Loss over Clean Targets           │
│    - Clean target labels: c_t ∈ {1, ..., K} (K-Means)  │
│    - Computed EXCLUSIVELY on masked time steps t ∈ M   │
│    - Loss: L = - ∑_{t ∈ M} log P(c_t | x̃)              │
└────────────────────────────────────────────────────────┘
```

### Architectural Configurations

| Parameter                          | WavLM Base             | WavLM Base+                                   | WavLM Large                                   |
| :--------------------------------- | :--------------------- | :-------------------------------------------- | :-------------------------------------------- |
| **Pre-training Data**        | 960h LibriSpeech       | 60k hrs (Libri-Light + VoxCeleb + GigaSpeech) | 60k hrs (Libri-Light + VoxCeleb + GigaSpeech) |
| **Transformer Layers**       | 12                     | 12                                            | 24                                            |
| **Hidden Dimension ($d$)** | 768                    | 768                                           | 1024                                          |
| **FFN Inner Dimension**      | 3072                   | 3072                                          | 4096                                          |
| **Attention Heads**          | 12                     | 12                                            | 16                                            |
| **Relative Pos Bias**        | Gated Relative         | Gated Relative                                | Gated Relative                                |
| **Total Parameters**         | **94.7 Million** | **94.7 Million**                        | **316.6 Million**                       |

---

# 3. Innovation 1: Gated Relative Position Bias (Math & Mechanics)

---

### A. Evolutionary Comparison of Positional Encodings in Speech AI

To understand why WavLM introduced **Gated Relative Position Bias**, we must examine the 4 generations of positional modeling in speech transformers and their respective failure modes on audio signals:

| Paradigm | Used In | Formulation | Core Strength | Fatal Flaw in Speech Processing |
| :--- | :--- | :--- | :--- | :--- |
| **1. Absolute Positional Embeddings** | Vanilla Transformer, BERT, ViT, AST | $x_i = z_i + p_i$, where $p_i = \text{Sinusoid}(i)$ | Simple, global index tracking | **Temporal Translation Variance**: A word spoken at $t=1.0\text{s}$ ($i=50$) vs $t=3.5\text{s}$ ($i=175$) has orthogonal positional vectors. The model cannot generalize that relative phoneme duration is invariant to where it appears in the audio stream. |
| **2. Convolutional Positional Encodings (PosConv)** | wav2vec 2.0, HuBERT | $p = \text{GELU}(\text{GroupNorm}(\text{Conv1D}(h, K=128, G=16)))$ | Translation invariant, captures local neighborhood | **Hard Receptive Field Ceiling**: Kernel size $K=128$ at 50Hz caps relative positioning at exactly $128 \times 20\text{ms} = 2.56\text{s}$. Beyond 2.56s, the network has zero geometric distance awareness, degrading long-form conversational turns and speaker diarization. |
| **3. Static Relative Position Bias** | T5, Transformer-XL, Shaw et al. | $\text{Attn}_{i,j} = \frac{q_i k_j^T}{\sqrt{d}} + r_{i-j}$ | Length-generalizable, smooth relative distance decay | **Content-Blind Inflexibility**: The distance decay $r_{i-j}$ is static and invariant to the audio acoustics. It enforces the exact same spatial penalty on transient plosives (which need tight 40ms local focus) and long vowels/speaker timbre (which need wide 3s global integration). |
| **4. Gated Relative Position Bias** | **WavLM** | $\text{Attn}_{i,j} = \frac{q_i k_j^T}{\sqrt{d}} + \sigma(W_g h_i + b_g) \cdot r_{i-j}$ | **Content-Adaptive Geometry**: Frame content $h_i$ dynamically scales local vs. global attention | **None**: Transient consonants dynamically scale up $g_i \to 1.0$ (enforcing sharp local bias), while sustained vowels scale down $g_i \to 0.0$ (allowing unrestrained global speaker timbre pooling). |

---

### B. Mathematical Formulation

In standard Multi-Head Self-Attention, query $q_i \in \mathbb{R}^{d_k}$ and key $k_j \in \mathbb{R}^{d_k}$ compute a dot-product affinity:

$$A_{i,j}^{\text{raw}} = \frac{q_i k_j^T}{\sqrt{d_k}}$$

WavLM introduces:
1. **Learnable Relative Distance Bias Table** $r \in \mathbb{R}^{2M - 1}$, mapping discrete relative offsets $\Delta_{i,j} = j - i \in [-M+1, M-1]$ to a continuous attention bias $r_{\Delta_{i,j}}$.
2. **Content-Dependent Dynamic Gating Function** $g_i$:

$$g_i = \sigma\left(W_g \cdot h_i + b_g\right)$$

where:
* $h_i \in \mathbb{R}^d$ is the input hidden state at query frame $i$.
* $W_g \in \mathbb{R}^{H \times d}$ is a learned weight projection (one per attention head $H$).
* $b_g \in \mathbb{R}^H$ is a learned bias scalar.
* $\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$ is the sigmoid activation function.

The final gated relative attention score $\alpha_{i,j}$ for each head is:

$$\tilde{R}_{i,j} = g_i \cdot r_{j - i}$$

$$S_{i,j} = \frac{q_i k_j^T}{\sqrt{d_k}} + \tilde{R}_{i,j}$$

$$\text{Attn\_Weight}_{i,j} = \frac{\exp(S_{i,j})}{\sum_{m=1}^T \exp(S_{i,m})}$$

---

### C. Step-by-Step Worked Numerical Example on Real Audio Segments

To see the exact mathematical impact, consider a **4-frame audio snippet ($T=4$, 80ms total at 16kHz/50Hz)** containing a transition from a silent onset to a plosive burst to a sustained vowel:

* **Frame 0 ($t=0\text{ms}$)**: Silence / Ambient noise background.
* **Frame 1 ($t=20\text{ms}$)**: Fast plosive consonant burst $[k]$ (transient duration $\approx 25\text{ms}$).
* **Frame 2 ($t=40\text{ms}$)**: Vowel onset $[\text{ae}]$ (periodic vocal fold vibration begins).
* **Frame 3 ($t=60\text{ms}$)**: Steady-state sustained vowel $[\text{ae}]$ (contains speaker pitch $F_0$ and formant timbre $F_1, F_2$).

Assume 1 attention head with dimension $d_k = 4$, $\sqrt{d_k} = 2.0$.

---

#### Step 1: Distance Matrix Calculation ($\Delta_{i,j} = j - i$)

$$\Delta = \begin{bmatrix}
0 - 0 & 1 - 0 & 2 - 0 & 3 - 0 \\
0 - 1 & 1 - 1 & 2 - 1 & 3 - 1 \\
0 - 2 & 1 - 2 & 2 - 2 & 3 - 2 \\
0 - 3 & 1 - 3 & 2 - 3 & 3 - 3
\end{bmatrix} = \begin{bmatrix}
0 & +1 & +2 & +3 \\
-1 & 0 & +1 & +2 \\
-2 & -1 & 0 & +1 \\
-3 & -2 & -1 & 0
\end{bmatrix}$$

---

#### Step 2: Static Learned Relative Position Bias Table Lookup ($r$)

Assume the pre-trained model learned the following distance decay profile (closer frames have higher affinity, distant frames are penalized):

| Relative Distance ($\Delta = j - i$) | -3 | -2 | -1 | 0 (Self) | +1 | +2 | +3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Learned Bias ($r_\Delta$)** | **-3.00** | **-1.80** | **-0.50** | **+1.20** | **-0.50** | **-1.80** | **-3.00** |

Mapping these values into the static relative position matrix $R$:

$$R = \begin{bmatrix}
+1.20 & -0.50 & -1.80 & -3.00 \\
-0.50 & +1.20 & -0.50 & -1.80 \\
-1.80 & -0.50 & +1.20 & -0.50 \\
-3.00 & -1.80 & -0.50 & +1.20
\end{bmatrix}$$

---

#### Step 3: Content-Dependent Gate Computation ($g_i = \sigma(W_g h_i + b_g)$)

The neural network inspects the acoustic properties of each query frame $h_i$:

1. **Frame 1 ($[k]$ plosive)**: Highly transient, local burst. The network decides it needs strict local focus:
   $$W_g h_1 + b_g = +2.20 \implies g_1 = \sigma(2.20) = \mathbf{0.90} \quad (\text{High Gate: Strongly enforces distance penalty})$$

2. **Frame 3 ($[\text{ae}]$ sustained vowel)**: Periodic, steady speaker formant. The network needs global context to measure speaker identity across the entire utterance:
   $$W_g h_3 + b_g = -2.20 \implies g_3 = \sigma(-2.20) = \mathbf{0.10} \quad (\text{Low Gate: Suppresses distance penalty})$$

Assume full gate vector across all 4 frames:
$$g = \begin{bmatrix} g_0 \\ g_1 \\ g_2 \\ g_3 \end{bmatrix} = \begin{bmatrix} 0.50 \\ \mathbf{0.90} \\ 0.40 \\ \mathbf{0.10} \end{bmatrix}$$

---

#### Step 4: Modulating the Relative Bias Matrix ($\tilde{R}_{i,j} = g_i \cdot R_{i,j}$)

Each row $i$ is multiplied by its scalar gate $g_i$:

$$\tilde{R} = \begin{bmatrix}
0.50 \times [+1.20, -0.50, -1.80, -3.00] \\
0.90 \times [-0.50, +1.20, -0.50, -1.80] \\
0.40 \times [-1.80, -0.50, +1.20, -0.50] \\
0.10 \times [-3.00, -1.80, -0.50, +1.20]
\end{bmatrix} = \begin{bmatrix}
+0.60 & -0.25 & -0.90 & -1.50 \\
\mathbf{-0.45} & \mathbf{+1.08} & \mathbf{-0.45} & \mathbf{-1.62} \\
-0.72 & -0.20 & +0.48 & -0.20 \\
\mathbf{-0.30} & \mathbf{-0.18} & \mathbf{-0.05} & \mathbf{+0.12}
\end{bmatrix}$$

Notice the critical behavior:
* For **Frame 1** (Plosive), the distant Frame 3 is penalized by **$-1.62$** (blocking remote distraction).
* For **Frame 3** (Vowel), the distant Frame 0 is penalized by only **$-0.30$** (allowing global cross-utterance integration).

---

#### Step 5: Attention Weights Comparison (Static Bias vs. WavLM Gated Bias)

Suppose raw acoustic dot products $\frac{q_i k_j^T}{\sqrt{d_k}}$ yield uniform baseline logits $A^{\text{raw}} = 0.0$ for all pairs. Let us observe the resulting Softmax attention distributions:

##### A. Query = Frame 1 (Transient Plosive $[k]$):
* **Raw Logits ($S_{1,:}$)** = $[-0.45, +1.08, -0.45, -1.62]$
* $\exp(S_{1,:}) = [0.638, 2.945, 0.638, 0.198]$, $\quad \sum = 4.419$
* **Attention Distribution**:
  $$\text{Attn}(1, :) = [\text{Fr}_0: \mathbf{14.4\%}, \quad \text{Fr}_1: \mathbf{66.6\%}, \quad \text{Fr}_2: \mathbf{14.4\%}, \quad \text{Fr}_3: \mathbf{4.5\%}]$$
  $\implies$ **81% of attention is tightly concentrated within a 20ms local window of the plosive.**

##### B. Query = Frame 3 (Sustained Vowel $[\text{ae}]$):
* **Raw Logits ($S_{3,:}$)** = $[-0.30, -0.18, -0.05, +0.12]$
* $\exp(S_{3,:}) = [0.741, 0.835, 0.951, 1.127]$, $\quad \sum = 3.654$
* **Attention Distribution**:
  $$\text{Attn}(3, :) = [\text{Fr}_0: \mathbf{20.3\%}, \quad \text{Fr}_1: \mathbf{22.8\%}, \quad \text{Fr}_2: \mathbf{26.0\%}, \quad \text{Fr}_3: \mathbf{30.8\%}]$$
  $\implies$ **Attention is distributed globally across all frames**, enabling the network to integrate speaker timbre, pitch fundamental frequency, and emotional prosody.

---

### D. Production PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedRelativePositionBias(nn.Module):
    """
    WavLM Gated Relative Position Bias Module.
    Dynamically scales learned relative position embeddings using content-dependent gating.
    """
    def __init__(self, embed_dim=768, num_heads=12, max_positions=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_positions = max_positions
        
        # Learnable relative position bias table for offsets in range [-(max_pos-1), +(max_pos-1)]
        # Shape: (2 * max_positions - 1, num_heads)
        self.rel_pos_embeddings = nn.Parameter(
            torch.randn(2 * max_positions - 1, num_heads) * 0.02
        )
        
        # Linear projection to compute content-dependent gate per head
        self.gate_proj = nn.Linear(embed_dim, num_heads)
        
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: Tensor of shape (Batch, Seq_Len, Embed_Dim)
        Returns:
            gated_bias: Tensor of shape (Batch, Num_Heads, Seq_Len, Seq_Len)
        """
        B, T, D = hidden_states.shape
        
        # 1. Generate Relative Distance Matrix: shape (T, T)
        range_vec = torch.arange(T, device=hidden_states.device)
        distance_mat = range_vec[None, :] - range_vec[:, None]  # Delta = j - i
        
        # 2. Clip distances to [-max_positions + 1, max_positions - 1]
        clipped_dist = torch.clamp(distance_mat, -self.max_positions + 1, self.max_positions - 1)
        index_mat = clipped_dist + self.max_positions - 1  # Shift to [0, 2*max_pos - 2]
        
        # 3. Lookup static relative bias: (T, T, num_heads) -> (num_heads, T, T)
        rel_bias = self.rel_pos_embeddings[index_mat].permute(2, 0, 1)  # (H, T, T)
        
        # 4. Compute Content Gate: g_i = sigmoid(W_g * h_i + b_g)
        # Shape: (B, T, H) -> (B, H, T, 1)
        gate = torch.sigmoid(self.gate_proj(hidden_states)).permute(0, 2, 1).unsqueeze(-1)
        
        # 5. Broadcast multiply: (B, H, T, 1) * (1, H, T, T) -> (B, H, T, T)
        gated_rel_bias = gate * rel_bias.unsqueeze(0)
        return gated_rel_bias


# Self-contained validation
if __name__ == "__main__":
    bias_module = GatedRelativePositionBias(embed_dim=768, num_heads=12, max_positions=512)
    sample_hidden = torch.randn(2, 200, 768)  # Batch=2, 200 frames (4 seconds @ 50Hz)
    gated_bias = bias_module(sample_hidden)
    
    assert gated_bias.shape == (2, 12, 200, 200), f"Unexpected shape {gated_bias.shape}"
    print("Gated Relative Position Bias shape verified successfully:", gated_bias.shape)
```

---

# 4. Innovation 2: Masked Speech Denoising & Overlap Simulation

Traditional SSL models take clean audio $x$, mask segments, and predict targets of $x$.
**WavLM introduces an on-the-fly acoustic corruption pipeline during pre-training**:

$$
\tilde{x} = x + \alpha \cdot n + \beta \cdot s
$$

where:

1. $x$: Primary clean speech signal.
2. $n$: Background acoustic noise sampled from AudioSet / MUSAN / DNS Challenge ($\text{SNR} \in [-5\text{dB}, 20\text{dB}]$ with probability $p_n = 0.5$).
3. $s$: Secondary overlapping speaker speech sampled randomly from the training corpus with probability $p_s = 0.2$.
4. $\alpha, \beta$: Random mixing energy scale coefficients.

### The Objective Paradox & Resolution

The corrupted audio $\tilde{x}$ is passed through the 1D CNN and masked. The Transformer context network must predict the **K-Means pseudo-phoneme tokens of the primary clean speech $x$**.

```text
[ Corrupted Audio: Speech A + Noise + Speech B ]
                       │
                       ▼
            [ CNN + Span Masking ]
                       │
                       ▼
          [ Transformer Context Network ]
                       │
                       ▼
  [ Predicted Target Units: Clean Speech A ONLY ]
```

### Why This Forces Full-Stack Capability:

* **For ASR**: The model learns to strip ambient noise and filter out background acoustic interference.
* **For Speaker ID & Diarization**: The model must track the dominant speaker's vocal characteristics amidst competing overlapping voices.
* **For Separation & Enhancement**: The internal layers learn implicit separation masks and subspace projections to isolate the target speaker.

---

# 5. Pre-Training Objective & Multi-Iteration Pipeline

WavLM uses the **HuBERT-style offline clustering iteration pipeline**:

### Iteration 1: MFCC Clustering

1. Extract 39-dimensional MFCC features (13 static + 13 $\Delta$ + 13 $\Delta\Delta$) from clean audio.
2. Train a K-Means clustering model with $K_1 = 512$ clusters on 10% of the dataset.
3. Assign each 20ms audio frame a discrete cluster ID $c_t \in \{1, \dots, 512\}$.
4. Pre-train WavLM Base for 400k steps using the Masked Denoising objective.

### Iteration 2: Latent Representation Clustering (WavLM Base+ & Large)

1. Extract the latent representations from the **6th and 9th Transformer layers** of the Iteration 1 model.
2. Train a second K-Means model with $K_2 = 1024$ clusters.
3. Pre-train the final WavLM model for 1 Million steps.

### Mathematical Pre-training Loss

Let $\mathcal{M}$ be the set of masked frame indices. The cross-entropy loss over masked frames is:

$$
\mathcal{L}_{\text{MLM}} = - \sum_{t \in \mathcal{M}} \log P\left(c_t \mid \tilde{z}_t, \mathcal{C}\right)
$$

$$
P\left(c \mid c_t\right) = \frac{\exp\left(\frac{h_t^T e_c}{\tau}\right)}{\sum_{k=1}^K \exp\left(\frac{h_t^T e_k}{\tau}\right)}
$$

where $h_t$ is the Transformer output embedding at frame $t$, $e_c$ is the learnable embedding for cluster $c$, and $\tau$ is the temperature parameter.

---

# 6. SUPERB Benchmark Dominance & Empirical Analysis

The **SUPERB (Speech Processing Universal PERformance Benchmark)** evaluates foundation speech models across 14 downstream tasks with **frozen backbones** (Linear Probing):

### Benchmark Comparison Table

| Model                      |  ASR (WER ↓)  |  PR (PER ↓)  |  SID (Acc ↑)  |  ASV (EER ↓)  |   SD (DER ↓)   |   ER (Acc ↑)   |  SS (SDRi ↑)  |  SE (PESQ ↑)  |
| :------------------------- | :------------: | :------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :------------: |
| **FBANK Baseline**   |     23.18     |     82.01     |      35.39      |      9.56%      |     10.05%     |     58.34%     |      9.23      |      2.55      |
| **Wav2Vec 2.0 Base** |      6.43      |      5.74      |      75.18      |      6.02%      |      6.08%      |     63.43%     |      10.45      |      2.62      |
| **HuBERT Base**      |      6.42      |      5.41      |      81.42      |      5.11%      |      5.88%      |     64.92%     |      10.60      |      2.65      |
| **WavLM Base**       |      6.21      |      4.84      |      84.51      |      4.67%      |      4.95%      |     65.94%     |      11.28      |      2.70      |
| **WavLM Base+**      |      5.59      |      4.41      | **89.43** | **3.89%** | **4.21%** | **68.20** | **11.64** | **2.78** |
| **WavLM Large**      | **3.44** | **3.22** | **94.67** | **1.85%** | **3.12%** | **70.62** | **12.55** | **2.92** |

* **ASR**: Automatic Speech Recognition (LibriSpeech test-clean)
* **PR**: Phoneme Recognition (TIMIT)
* **SID**: Speaker Identification (VoxCeleb1)
* **ASV**: Automatic Speaker Verification (VoxCeleb1 test)
* **SD**: Speaker Diarization (LibriCSS)
* **ER**: Emotion Recognition (IEMOCAP)
* **SS**: Speech Separation (WSJ0-2mix)
* **SE**: Speech Enhancement (VoiceBank-DEMAND)

---

# 7. Full Standalone PyTorch Implementation

Below is a complete, standalone PyTorch implementation of the **WavLM Model Architecture**, including the 1D Temporal Convolutional Feature Extractor, Gated Relative Position Bias Self-Attention, and the Downstream Classification Head.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================================
# 1. 1D Temporal Convolutional Feature Extractor (7 Conv Layers)
# ====================================================================
class WavLMFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, conv_dim=512):
        super().__init__()
        # Strides: (5, 2, 2, 2, 2, 2, 2) -> Total downsampling factor = 320 (20ms stride @ 16kHz)
        # Kernels: (10, 3, 3, 3, 3, 2, 2)
        conv_layers = []
        kernel_sizes = [10, 3, 3, 3, 3, 2, 2]
        strides = [5, 2, 2, 2, 2, 2, 2]
      
        in_d = in_channels
        for k, s in zip(kernel_sizes, strides):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_d, conv_dim, kernel_size=k, stride=s, bias=False),
                    nn.GroupNorm(num_groups=conv_dim, num_channels=conv_dim, affine=True),
                    nn.GELU()
                )
            )
            in_d = conv_dim
          
        self.layers = nn.ModuleList(conv_layers)
      
    def forward(self, x):
        # Input shape: (Batch, Samples) -> (Batch, 1, Samples)
        if x.dim() == 2:
            x = x.unsqueeze(1)
          
        for layer in self.layers:
            x = layer(x)
          
        # Output shape: (Batch, 512, Time_Frames) -> (Batch, Time_Frames, 512)
        return x.transpose(1, 2)


# ====================================================================
# 2. Multi-Head Self-Attention with Gated Relative Position Bias
# ====================================================================
class WavLMGatedRelPosAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, max_positions=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_positions = max_positions
      
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
      
        # Gated Relative Position Bias
        self.rel_pos_table = nn.Parameter(torch.randn(2 * max_positions - 1, num_heads) * 0.02)
        self.gate_proj = nn.Linear(embed_dim, num_heads)
      
    def forward(self, x):
        B, T, D = x.shape
      
        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
      
        # Standard Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, H, T, T)
      
        # Compute Gated Relative Position Bias
        range_vec = torch.arange(T, device=x.device)
        dist_mat = range_vec[None, :] - range_vec[:, None]
        clipped_dist = torch.clamp(dist_mat, -self.max_positions + 1, self.max_positions - 1)
        indices = clipped_dist + self.max_positions - 1
        rel_bias = self.rel_pos_table[indices].permute(2, 0, 1)  # (H, T, T)
      
        gate = torch.sigmoid(self.gate_proj(x)).permute(0, 2, 1).unsqueeze(-1)  # (B, H, T, 1)
        gated_rel_bias = gate * rel_bias.unsqueeze(0)  # (B, H, T, T)
      
        scores = scores + gated_rel_bias
        attn_weights = F.softmax(scores, dim=-1)
      
        context = torch.matmul(attn_weights, v) # (B, H, T, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(context)


# ====================================================================
# 3. WavLM Transformer Encoder Block
# ====================================================================
class WavLMEncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, ffn_dim=3072, num_heads=12):
        super().__init__()
        self.attn = WavLMGatedRelPosAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
      
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
      
    def forward(self, x):
        # Pre-LN Transformer Architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ====================================================================
# 4. Full WavLM Model for Sequence Classification (SER / SID)
# ====================================================================
class WavLMForSequenceClassification(nn.Module):
    def __init__(self, num_labels=8, embed_dim=768, num_layers=12, num_heads=12):
        super().__init__()
        self.feature_extractor = WavLMFeatureExtractor(in_channels=1, conv_dim=512)
        self.feature_projection = nn.Linear(512, embed_dim)
      
        self.layers = nn.ModuleList([
            WavLMEncoderLayer(embed_dim=embed_dim, ffn_dim=3072, num_heads=num_heads)
            for _ in range(num_layers)
        ])
      
        # Classification Head: Mean Pooling + Linear Projector + Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )
      
    def forward(self, x):
        # 1. Feature Extraction: (B, 64000) -> (B, 199, 512)
        features = self.feature_extractor(x)
        h = self.feature_projection(features)
      
        # 2. Transformer Context Network
        for layer in self.layers:
            h = layer(h)
          
        # 3. Global Mean Pooling over time dimension
        pooled = torch.mean(h, dim=1)  # (B, 768)
      
        # 4. Logits
        logits = self.classifier(pooled) # (B, num_labels)
        return logits


# Quick Verification Test
if __name__ == "__main__":
    model = WavLMForSequenceClassification(num_labels=8)
    dummy_wav = torch.randn(2, 64000)  # 2 samples of 4s audio @ 16kHz
    logits = model(dummy_wav)
    print("WavLM Output Logits Shape:", logits.shape)  # Expected: torch.Size([2, 8])
```

---

# 8. Practical Fine-Tuning Guide & Real-World Use Cases

### 1. Speech Emotion Recognition (SER)

```python
from transformers import AutoFeatureExtractor, WavLMForSequenceClassification

# Load HuggingFace pre-trained checkpoint
model = WavLMForSequenceClassification.from_pretrained(
    "microsoft/wavlm-base-plus",
    num_labels=8
)

# Freeze CNN Feature Extractor (standard best practice)
model.freeze_feature_extractor()

# Use layer-wise learning rate decay
optimizer = torch.optim.AdamW([
    {"params": model.wavlm.parameters(), "lr": 2e-5},
    {"params": model.classifier.parameters(), "lr": 1e-3}
])
```

### 2. Weighted Layer Sum Strategy (Maximizing SUPERB Score)

Rather than taking only the last Transformer layer, extract all $L$ layers and learn a softmax-weighted sum:

$$
h_{\text{repr}} = \sum_{l=0}^L w_l \cdot h^{(l)}, \quad \text{where } \sum w_l = 1
$$

* **Layers 1–4**: Best for Speech Enhancement & Denoising.
* **Layers 5–8**: Best for Speaker Identification & Diarization.
* **Layers 9–12**: Best for Phoneme Recognition & ASR.

---

### Summary Checklist for Engineering Deployments

* **Audio Sampling Rate**: Always resample audio to **16,000 Hz**.
* **Model Selection**:
  * Use `microsoft/wavlm-base` for lightweight ASR/SER applications on edge GPUs.
  * Use `microsoft/wavlm-base-plus` for robust real-world noisy environments and multi-speaker applications.
  * Use `microsoft/wavlm-large` for state-of-the-art competitive benchmarks.
