# HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units

## Paper Information

- **Title:** HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units
- **Authors:** Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed (Meta AI / CMU)
- **Year / ArXiv ID:** 2021 | arXiv:2106.07447v1
- **Code:** [fairseq/examples/hubert](https://github.com/pytorch/fairseq/tree/master/examples/hubert)

---

## 1. Executive Summary & Core Motivation

In speech processing, self-supervised learning faces three primary hurdles compared to Computer Vision (CV) and Natural Language Processing (NLP):

1. **Continuous & Unsegmented:** Speech is a continuous acoustic waveform with no explicit token/word/phoneme boundaries.
2. **No Pre-existing Discrete Lexicon:** Unlike text (which has words/subwords tokens) or image classification (instance tokens), speech lacks a predefined discrete vocabulary for self-supervised prediction targets during pre-training.
3. **Multiple Acoustic Units per Utterance:** Each utterance contains complex mixtures of phonetic information, speaker timbre, prosody, and background noise.

**HuBERT (Hidden-Unit BERT)** solves this by decoupling **acoustic unit discovery** from **masked prediction representation learning**. It uses an offline clustering step (e.g., $k$-means) to generate discrete target labels for each frame, and then trains a BERT-like Transformer to predict these cluster IDs only for masked audio frames.

---

## 2. Model Architecture & Workflow

### A. Architecture

1. **Convolutional Waveform Encoder:**
   - 7 temporal convolution blocks (kernel sizes [10, 3, 3, 3, 3, 2, 2], strides [5, 2, 2, 2, 2, 2, 2], channel size 512).
   - Total downsampling factor of **320x** (transforms 16 kHz raw audio into 20ms feature representations at 50 Hz).
2. **BERT / Transformer Encoder:**
   - Standard Transformer encoder stack (Base: 12 layers, 95M params; Large: 24 layers, 317M params; X-Large: 48 layers, 964M params / ~1B).
3. **Projection & Prediction Head:**
   - Projects Transformer hidden states to compute cosine similarity against learned code embeddings for each discrete cluster ID.

```
       [Raw Audio Waveform (16 kHz)]
                     │
                     ▼
       ┌───────────────────────────┐
       │   7-layer CNN Encoder     │ (Downsamples by 320x -> 20ms frames)
       └─────────────┬─────────────┘
                     │
         [Continuous Audio Frames]
                     │
                     ├───────────────►  [ Offline Unit Discovery (k-means) ]
                     │                                 │
            [Span Masking (p=8%)]                      │
                     │                                 ▼
                     ▼                       [ Target Discrete Units ]
       ┌───────────────────────────┐         (Cluster IDs: z_1, z_2, ...)
       │ Transformer Encoder (BERT)│                   │
       └─────────────┬─────────────┘                   │
                     │                                 │
             [Hidden States o_t]                       │
                     │                                 │
                     ▼                                 │
        [Cosine Sim Projection Head] ◄─────────────────┘
                     │
             [ Cross-Entropy Loss on Masked Frames ONLY ]
```

---

## 3. Key Methodological Innovations

### 1. Offline Acoustic Clustering

- **Iteration 1:** Runs simple $k$-means clustering ($K=100$) on standard 39-D MFCC features (13 MFCCs + $\Delta$ + $\Delta\Delta$).
- **Iteration 2 & Beyond (Iterative Refinement):** Extracts intermediate representations (e.g., 6th layer of Base model) from the pre-trained model and clusters them ($K=500$) to create significantly cleaner, phone-correlated pseudo-labels.

### 2. Masked Cross-Entropy Loss ($\alpha = 1.0$)

- A critical finding of HuBERT is that computing cross-entropy loss **exclusively on the masked spans** forces the model to act as both an **acoustic model** (processing unmasked context) and a **language model** (predicting masked content from context).
- Computing loss on unmasked frames causes the model to simply memorize noisy cluster assignments, severely harming generalization.

### 3. Cluster Ensembling & Product Quantization

- Pre-training with ensembles of cluster teachers (e.g., $K=\{50, 100, 500\}$ or Product $k$-means on sub-vectors) provides complementary granularity and improves representation richness.

---

## 4. Key Experimental Results

| Model / Fine-tuning Data               |   10 min Labeled   |    1 hr Labeled    |    10 hr Labeled    |   100 hr Labeled   | 960 hr (Full LibriSpeech) |
| :------------------------------------- | :-----------------: | :-----------------: | :-----------------: | :-----------------: | :-----------------------: |
| **wav2vec 2.0 Large** (LL-60k)   |      4.8 / 8.2      |      2.9 / 5.8      |      2.6 / 4.9      |      2.0 / 4.0      |         1.8 / 3.3         |
| **HuBERT Large** (LL-60k)        |      4.7 / 7.6      |      2.9 / 5.4      |      2.4 / 4.6      |      2.1 / 3.9      |         1.9 / 3.3         |
| **HuBERT X-Large (1B)** (LL-60k) | **4.6 / 6.8** | **2.8 / 4.8** | **2.3 / 4.0** | **1.9 / 3.5** |    **1.8 / 2.9**    |

*(WER reported on LibriSpeech test-clean / test-other)*

- **Ultra-low resource:** With only **10 minutes** of labeled data, HuBERT X-Large achieves **4.6% WER** (test-clean) and **6.8% WER** (test-other).
- **Noisy & Challenging conditions:** Up to **13% to 19% relative WER reduction** on `test-other` / `dev-other` over wav2vec 2.0 Large.
- **Simpler Objective:** Unlike wav2vec 2.0, HuBERT avoids Gumbel-Softmax temperature annealing schedules, contrastive negative sampling buffers, and auxiliary diversity loss terms.

---

## 5. Comparison: HuBERT vs. wav2vec 2.0 vs. DiscreteBERT

| Feature                     | HuBERT                                  | wav2vec 2.0                            | DiscreteBERT              |
| :-------------------------- | :-------------------------------------- | :------------------------------------- | :------------------------ |
| **Input**             | Continuous raw waveform                 | Continuous raw waveform                | Discrete quantized tokens |
| **Target Units**      | Offline$k$-means clusters (iterative) | Online Gumbel-Softmax codebook         | vq-wav2vec discrete units |
| **Pre-training Loss** | Cross-entropy on masked frames          | Contrastive loss + Diversity loss      | Masked language modeling  |
| **Refinement**        | Iterative re-clustering on latents      | Single-stage end-to-end                | Fixed single teacher      |
| **Stability**         | Highly stable, standard cross-entropy   | Requires careful temperature annealing | Stable                    |
