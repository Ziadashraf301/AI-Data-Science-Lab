# wav2vec 2.0: Deep Technical & Mathematical Study Guide

> **Paper Title:** *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*  
> **Authors:** Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli (Meta AI, 2020)

---

## 1. High-Level Concept & Conceptual Intuition

### The Fundamental Problem in Speech Recognition
Traditional Automatic Speech Recognition (ASR) relied on **supervised learning**, requiring thousands of hours of parallel audio-to-text pairs (Audio X, Text Y). Out of ~7,000 languages spoken worldwide, only a tiny fraction have sufficient annotated audio datasets.

### The Self-Supervised Paradigm Shift
Human infants do not learn language by reading transcribed subtitles. They listen to tens of thousands of hours of raw audio, discovering phonemes, pitch, and acoustic structures **unsupervised**. 

**wav2vec 2.0** applies this exact mechanism:
1. **Stage 1 (Unsupervised Pre-training):** Train on thousands of hours of raw, unlabeled audio X. The model learns to predict masked segments of audio by building internal discrete acoustic codebooks (pseudo-phonemes).
2. **Stage 2 (Supervised Fine-tuning):** Add a simple linear projection layer on top of the contextual representations and train with **Connectionist Temporal Classification (CTC)** loss on a tiny amount of labeled text (e.g., 10 minutes to 1 hour).

---

## 2. Complete Architectural Pipeline & Dataflow

Below is the complete dataflow from raw waveform audio to downstream character prediction:

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
│ - Probability p = 0.065  │  │ Z ──► Q                  │
│ - Span length M = 10     │  │ - G = 2 Codebooks        │
│ - ~49% total time steps  │  │ - V = 320 Entries each   │
│   replaced by mask vector│  │ - Gumbel-Softmax selection│
└────────────┬─────────────┘  └────────────┬─────────────┘
             │                             │
             ▼                             │
┌──────────────────────────┐               │
│ 2. Transformer Context   │               │
│    Network g: Z ──► C    │               │
│ - 12 (BASE) / 24 (LARGE) │               │
│   blocks                 │               │
│ - Relative Positional    │               │
│   Embeddings (1D Conv)   │               │
│ - Output: Context c_t    │               │
└────────────┬─────────────┘               │
             │                             │
             ▼                             ▼
┌────────────────────────────────────────────────────────┐
│ 3. Pre-Training Loss Calculation                       │
│    - Contrastive Loss L_m (distinguish q_t from K=100  │
│      distractors using Cosine Similarity)              │
│    - Codebook Diversity Penalty L_d                    │
│    - Total Pre-training Loss: L = L_m + (alpha * L_d)  │
└────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Mathematical Formulation & Step-by-Step Numerical Example

### Step A: Feature Encoder f: X ──► Z
- **Input:** Raw audio signal X.
- **Encoder:** A 7-layer temporal 1D Convolutional Neural Network.
- **Strides:** (5, 2, 2, 2, 2, 2, 2) ➔ Total Stride = 5 * 2^6 = 320.
- For 16,000 Hz audio, each feature vector z_t (dimension d = 512) represents:
  
  **Frame Duration = 320 samples / 16,000 samples/sec = 0.020 sec = 20 ms of audio stride**

---

### Step B: Quantization Module Z ──► Q (Product Quantization & Gumbel-Softmax)

To create discrete acoustic targets without human phoneme annotations, z_t is mapped to a discrete target vector q_t.

#### 1. Codebook Layout:
- G codebooks (groups), typically **G = 2**.
- Each codebook contains V entries (codewords), typically **V = 320**.
- Each entry e_(g,v) has dimension d / G (for BASE model: 512 / 2 = 256).
- Total potential unique discrete representations: **V^G = 320^2 = 102,400 combinations**.

#### 2. Differentiable Selection via Gumbel-Softmax:
To pick discrete entries while allowing gradient backpropagation:
1. Map latent vector z_t via linear layer to logits l in R^(G x V).
2. The probability p_(g,v) of choosing the v-th entry in group g is:

   **p_(g,v) = exp( (l_(g,v) + n_v) / tau ) / [ sum_(k=1..V) exp( (l_(g,k) + n_k) / tau ) ]**

   where:
   - n_v = -log(-log(u)) with u sampled from Uniform(0, 1) (Gumbel noise).
   - tau is the temperature parameter annealed from tau_max = 2.0 down to tau_min = 0.5.

3. During forward pass, choose codeword index v* = argmax_v p_(g,v).
4. Concatenate chosen vectors from each group:
   
   **e_1 in R^256,  e_2 in R^256 ➔ [e_1; e_2] in R^512**

5. Apply linear transformation R^d ──► R^f to obtain final target vector q_t in R^f.

---

### Step C: The Pre-Training Objective Functions

#### 1. Contrastive Loss (L_m)
For a masked time step t, the context vector c_t produced by the Transformer must identify the true target q_t among K = 100 distractor targets q_tilde in Q_t sampled from other masked time steps in the same utterance:

**L_m = -log [ exp( sim(c_t, q_t) / kappa ) / sum_(q_tilde in Q_t) exp( sim(c_t, q_tilde) / kappa ) ]**

where:
- **sim(a, b) = (a . b) / (||a|| * ||b||)** is the Cosine Similarity.
- **kappa = 0.1** is the temperature hyperparameter.

---

### 🔢 Concrete Numerical Example of Contrastive Loss

Imagine a simplified setup:
- Temperature **kappa = 0.1**.
- Target q_t and context c_t vectors are normalized.
- Number of distractors **K = 2** (total candidates = 3: 1 True Target + 2 Distractors).

#### Given Cosine Similarities:
1. **True Target (q_t):** sim(c_t, q_t) = **0.80**
2. **Distractor 1 (q_tilde_1):** sim(c_t, q_tilde_1) = **0.20**
3. **Distractor 2 (q_tilde_2):** sim(c_t, q_tilde_2) = **-0.10**

#### Step-by-Step Loss Calculation:

1. **Divide similarities by temperature kappa = 0.1:**
   - Score(True Target) = 0.80 / 0.1 = **8.0**
   - Score(Distractor 1) = 0.20 / 0.1 = **2.0**
   - Score(Distractor 2) = -0.10 / 0.1 = **-1.0**

2. **Exponentiate scores:**
   - exp(8.0) ≈ **2980.958**
   - exp(2.0) ≈ **7.389**
   - exp(-1.0) ≈ **0.368**

3. **Sum denominator:**
   - Denominator = 2980.958 + 7.389 + 0.368 = **2988.715**

4. **Compute Softmax Probability for True Target:**
   - P(True Target) = 2980.958 / 2988.715 ≈ **0.9974** (99.74%)

5. **Compute Contrastive Loss L_m:**
   - L_m = -log(0.9974) ≈ **0.0026**

*Interpretation:* Because the similarity of the true target (0.80) was much higher than the distractors, the model probability is 99.74%, leading to a very low loss value of 0.0026.

---

#### 2. Codebook Diversity Penalty (L_d)
If unconstrained, the model might fall into a degenerate state where it reuses only 5 or 10 codebook entries, ignoring the remaining 315. To prevent codebook collapse, L_d maximizes the entropy of the average Gumbel-Softmax probabilities p_bar_(g,v) across a batch:

**L_d = [ 1 / (G * V) ] * sum_(g=1..G) [ sum_(v=1..V) p_bar_(g,v) * log(p_bar_(g,v)) ]**

- **Total Combined Loss:**
  
  **L = L_m + (alpha * L_d)**   *(where alpha = 0.1)*

---

## 4. Key Experimental Results & Benchmarks

| Metric / Setting | Baseline / Previous SOTA | wav2vec 2.0 (BASE) | wav2vec 2.0 (LARGE) |
| :--- | :--- | :--- | :--- |
| **Unlabeled Pre-training Data** | N/A | 960h (LibriSpeech) | 53k hours (LibriVox) |
| **WER (10 Minutes Labeled Data)** | 25.2% (Discrete BERT) | 9.1% (Clean) / 15.6% (Other) | **4.8% (Clean) / 8.2% (Other)** |
| **WER (1 Hour Labeled Data)** | 9.0% (Discrete BERT) | 5.5% (Clean) / 11.3% (Other) | **2.9% (Clean) / 5.8% (Other)** |
| **WER (Full 960h Labeled Data)** | 2.1% (ContextNet) | 2.1% (Clean) / 4.8% (Other) | **1.8% (Clean) / 3.3% (Other)** |

---

## 5. Summary & Key Takeaways for Machine Learning Practitioners

1. **Discrete vs. Continuous Targets:**
   Pre-training with quantized target units (q_t) works significantly better than continuous targets because continuous targets allow the model to cheat by capturing background noise or speaker-specific pitch instead of phone-level linguistic properties.
2. **Context Network Input is Continuous:**
   While the target q_t is discretized, the input fed into the Transformer context network remains continuous z_t. This allows the Transformer to retain full detail across sequences.
3. **Extreme Low-Resource Feasibility:**
   Pre-training on raw audio reduces the reliance on manual transcriptions by up to **100x to 1000x**, enabling high-accuracy speech models for low-resource languages worldwide.
