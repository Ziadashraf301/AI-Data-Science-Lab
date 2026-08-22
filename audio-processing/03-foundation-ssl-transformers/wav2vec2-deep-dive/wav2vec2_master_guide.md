# 📖 wav2vec 2.0: Complete Master Technical & Implementation Reference

> **Paper Title:** *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*  
> **Authors:** Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli (Meta AI, 2020)

---

# TABLE OF CONTENTS
1. [Executive Summary & High-Level Intuition](#1-executive-summary--high-level-intuition)
2. [Complete System Architecture & Dataflow](#2-complete-system-architecture--dataflow)
3. [Convolutional Feature Encoder (Shape Breakdown & Code)](#3-convolutional-feature-encoder-shape-breakdown--code)
4. [Quantization Module & Product Quantization (Math & Walkthrough)](#4-quantization-module--product-quantization-math--walkthrough)
5. [Span Masking Engine & Transformer Context Encoder](#5-span-masking-engine--transformer-context-encoder)
6. [Full Wav2Vec2Model Class & Method-by-Method Breakdown](#6-full-wav2vec2model-class--method-by-method-breakdown)
7. [Pre-Training Loss Functions & Worked Numerical Example](#7-pre-training-loss-functions--worked-numerical-example)
8. [Experimental Results & Benchmarks](#8-experimental-results--benchmarks)
9. [Practical Fine-Tuning Guide & Real-World Use Cases](#9-practical-fine-tuning-guide--real-world-use-cases)

---

# 1. Executive Summary & High-Level Intuition

### The Fundamental Problem in Traditional Speech Recognition (ASR)
Traditional Automatic Speech Recognition (ASR) relied on **supervised learning**, requiring thousands of hours of parallel audio-to-text pairs $(\mathcal{X}, Y)$. Out of $\sim 7,000$ languages spoken worldwide, less than 1% have sufficient annotated audio datasets.

### The Self-Supervised Paradigm Shift
Human infants do not learn language by reading transcribed subtitles. They listen to tens of thousands of hours of raw audio, discovering phonemes, pitch, and acoustic structures **unsupervised**. 

**wav2vec 2.0** applies this exact mechanism:
1. **Stage 1 (Unsupervised Pre-training):** Train on thousands of hours of raw, unlabeled audio $\mathcal{X}$. The model learns to predict masked segments of audio by building internal discrete acoustic codebooks (pseudo-phonemes).
2. **Stage 2 (Supervised Fine-tuning):** Add a simple linear projection layer on top of the contextual representations and train with **Connectionist Temporal Classification (CTC)** loss on a tiny amount of labeled text (e.g., 10 minutes to 1 hour).

---

# 2. Complete System Architecture & Dataflow

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

# 3. Convolutional Feature Encoder (Shape Breakdown & Code)

The 1D Convolutional Feature Encoder takes raw 16kHz audio and downsamples it into a sequence of feature vectors $z_t$.

### 1D Convolution Output Length Formula
For any 1D Conv layer with input length $L_{in}$, kernel size $K$, stride $S$, and no padding ($P=0$):

$$L_{out} = \left\lfloor \frac{L_{in} - K}{S} \right\rfloor + 1$$

PyTorch Conv1D Weight Tensor Format: `(out_channels, in_channels, kernel_size)`

### Step-by-Step Shape Trace (Input: 2 seconds of 16kHz audio = 32,000 samples)
Input Tensor Shape: `(Batch=1, Channels=1, Length=32000)`

| Layer | Kernel ($K$) | Stride ($S$) | Computation | Output Length ($L_{out}$) | Conv Weight Shape | Output Tensor Shape `(B, C, L)` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Input Audio** | - | - | - | 32,000 | - | `(1, 1, 32000)` |
| **Layer 1** | 10 | 5 | $\lfloor (32000 - 10)/5 \rfloor + 1$ | **6,400** | `(512, 1, 10)` | `(1, 512, 6400)` |
| **Layer 2** | 3 | 2 | $\lfloor (6400 - 3)/2 \rfloor + 1$ | **3,199** | `(512, 512, 3)` | `(1, 512, 3199)` |
| **Layer 3** | 3 | 2 | $\lfloor (3199 - 3)/2 \rfloor + 1$ | **1,599** | `(512, 512, 3)` | `(1, 512, 1599)` |
| **Layer 4** | 3 | 2 | $\lfloor (1599 - 3)/2 \rfloor + 1$ | **799** | `(512, 512, 3)` | `(1, 512, 799)` |
| **Layer 5** | 3 | 2 | $\lfloor (799 - 3)/2 \rfloor + 1$ | **399** | `(512, 512, 3)` | `(1, 512, 399)` |
| **Layer 6** | 2 | 2 | $\lfloor (399 - 2)/2 \rfloor + 1$ | **199** | `(512, 512, 2)` | `(1, 512, 199)` |
| **Layer 7** | 2 | 2 | $\lfloor (199 - 2)/2 \rfloor + 1$ | **99** | `(512, 512, 2)` | `(1, 512, 99)` |

### Reshape to Sequence Format
After Layer 7, transpose channels and time steps:
$$\text{Shape: } (1, 512, 99) \xrightarrow{\text{transpose}(1, 2)} \mathbf{(1, 99, 512)}$$
Total stride factor $= 5 \times 2^6 = 320$. Each step represents $\frac{320}{16000} = \mathbf{20\text{ ms}}$ of audio.

### Python Implementation (`ConvFeatureEncoder`)
```python
class ConvFeatureEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=512, verbose=False):
        super().__init__()
        self.verbose = verbose
        
        conv_params = [
            (512, 10, 5),  # Layer 1 Weight -> (512, 1, 10)
            (512,  3, 2),  # Layer 2 Weight -> (512, 512, 3)
            (512,  3, 2),  # Layer 3 Weight -> (512, 512, 3)
            (512,  3, 2),  # Layer 4 Weight -> (512, 512, 3)
            (512,  3, 2),  # Layer 5 Weight -> (512, 512, 3)
            (512,  2, 2),  # Layer 6 Weight -> (512, 512, 2)
            (512,  2, 2),  # Layer 7 Weight -> (512, 512, 2)
        ]
        
        self.conv_layers = nn.ModuleList()
        curr_in = in_channels
        for out_ch, k_size, stride in conv_params:
            conv = nn.Conv1d(curr_in, out_ch, kernel_size=k_size, stride=stride, bias=False)
            block = nn.Sequential(conv, nn.Dropout(p=0.0), nn.GELU())
            self.conv_layers.append(block)
            curr_in = out_ch
            
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        if self.verbose:
            print(f"Input Audio Shape: {list(x.shape)}")
            
        for idx, block in enumerate(self.conv_layers):
            conv_layer = block[0]
            weight_shape = list(conv_layer.weight.shape)
            x = block(x)
            if self.verbose:
                print(f"Conv Layer {idx+1} | Weight: {weight_shape} | Output: {list(x.shape)}")
            
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return x  # Shape: (Batch, T, 512)
```

---

# 4. Quantization Module & Product Quantization (Math & Walkthrough)

The Quantization Module maps continuous latent features $z_t \in \mathbb{R}^{512}$ to discrete codebook vectors $q_t \in \mathbb{R}^{512}$.

### Where does $320 \times 320 = 102,400$ combinations come from?
- **Groups (Codebooks):** $G = 2$
- **Entries per Group:** $V = 320$
- Group 1 selects **1 entry out of 320** ($e_{1, i}$).
- Group 2 selects **1 entry out of 320** ($e_{2, j}$).

$$q_t = [e_{1, i} \,\, ; \,\, e_{2, j}] \in \mathbb{R}^{512}$$

$$\text{Total Combinations} = V_1 \times V_2 = 320 \times 320 = \mathbf{102,400}$$

#### 💡 2D Grid Index Formula Mapping:
Converting 2D selection `(Row i, Col j)` to a 1D ID index:

$$\text{Unique Combination ID} = (i \times 320) + j$$

For Example: Group 1 Entry #12 and Group 2 Entry #3:
$$\text{ID} = (12 \times 320) + 3 = 3840 + 3 = \mathbf{3,843 \text{ out of 102,400}}$$

#### Real-World Audio Intuition Example (Vowels vs. Prosody)
- **Group 1 ($V=320$):** Learns Phonetic Sound Type (Entry #12 = Open Vowel `/a/`).
- **Group 2 ($V=320$):** Learns Prosody & Speaker Features (Entry #3 = High Pitch Voice).
- Combination for a High Pitch `/a/` vowel: $q_t = [e_{1, 12} \, ; \, e_{2, 3}]$ (Index **#3,843** out of 102,400).

### Python Implementation (`GumbelVectorQuantizer`)
```python
class GumbelVectorQuantizer(nn.Module):
    def __init__(self, in_dim=512, num_groups=2, num_vars=320, vq_dim=512, temp=(2.0, 0.5, 0.999995)):
        super().__init__()
        self.G = num_groups
        self.V = num_vars
        self.vq_dim = vq_dim
        self.entry_dim = vq_dim // self.G  # 512 / 2 = 256
        
        self.weight_proj = nn.Linear(in_dim, self.G * self.V)
        self.vars = nn.Parameter(torch.FloatTensor(1, self.G * self.V, self.entry_dim))
        nn.init.uniform_(self.vars, -1.0 / math.sqrt(self.entry_dim), 1.0 / math.sqrt(self.entry_dim))
        self.curr_temp = temp[0]

    def forward(self, x):
        B, T, D = x.size()
        logits = self.weight_proj(x)
        logits_grouped = logits.view(B, T, self.G, self.V)
        
        if self.training:
            gumbel_noise = -torch.empty_like(logits_grouped).exponential_().log()
            soft_probs = F.softmax((logits_grouped + gumbel_noise) / self.curr_temp, dim=-1)
        else:
            soft_probs = F.softmax(logits_grouped / self.curr_temp, dim=-1)
            
        _, max_idx = soft_probs.max(dim=-1)
        hard_onehot = torch.zeros_like(logits_grouped).scatter_(-1, max_idx.unsqueeze(-1), 1.0)
        
        # Straight-Through Estimator Trick
        code_probs = hard_onehot - soft_probs.detach() + soft_probs
        
        vars_reshaped = self.vars.view(self.G, self.V, self.entry_dim)
        selected_entries = torch.einsum('btgv,gvd->btgd', code_probs, vars_reshaped)
        q_targets = selected_entries.reshape(B, T, self.G * self.entry_dim)
        
        avg_probs = soft_probs.mean(dim=(0, 1))
        prob_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)).sum()
        return q_targets, prob_perplexity, avg_probs
```

---

# 5. Span Masking Engine & Transformer Context Encoder

### Span Masking Mechanism
- Masks **spans of 10 consecutive time steps** ($\approx 200\text{ms}$ of audio).
- Overlapping spans mask approximately **49% of all time steps**.
- Masked frames are replaced by a trainable vector `mask_emb` $\in \mathbb{R}^{512}$.

### Transformer Context Encoder
- 12 layers (BASE) or 24 layers (LARGE) of Transformer blocks.
- Uses self-attention to build contextual representations $c_t$ for both masked and unmasked frames.

---

# 6. Full Wav2Vec2Model Class & Method-by-Method Breakdown

```python
class Wav2Vec2Model(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, num_groups=2, num_vars=320, verbose_encoder=True):
        super().__init__()
        self.feature_extractor = ConvFeatureEncoder(in_channels=1, embed_dim=embed_dim, verbose=verbose_encoder)
        self.quantizer = GumbelVectorQuantizer(in_dim=embed_dim, num_groups=num_groups, num_vars=num_vars, vq_dim=embed_dim)
        
        self.mask_emb = nn.Parameter(torch.FloatTensor(embed_dim))
        nn.init.uniform_(self.mask_emb)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.project_q = nn.Linear(embed_dim, embed_dim)

    def apply_masking(self, z, mask_prob=0.065, mask_length=10):
        B, T, D = z.size()
        mask_indices = torch.zeros(B, T, dtype=torch.bool, device=z.device)
        for b in range(B):
            num_spans = int(mask_prob * T)
            if num_spans > 0:
                starts = torch.randperm(max(1, T - mask_length))[:num_spans]
                for s in starts:
                    mask_indices[b, s : s + mask_length] = True
        z_masked = z.clone()
        z_masked[mask_indices] = self.mask_emb
        return z_masked, mask_indices

    def compute_contrastive_loss(self, c, q, mask_indices, num_negatives=10, temp=0.1):
        if not mask_indices.any():
            return torch.tensor(0.0, device=c.device)
            
        c_masked = c[mask_indices]  # (N_masked, 512)
        q_masked = q[mask_indices]  # (N_masked, 512)
        
        c_norm = F.normalize(c_masked, dim=-1)
        q_norm = F.normalize(q_masked, dim=-1)
        N_masked = c_masked.size(0)
        
        negs = []
        for i in range(N_masked):
            other_indices = [idx for idx in range(N_masked) if idx != i]
            if len(other_indices) >= num_negatives:
                sampled_idx = torch.tensor(other_indices)[torch.randperm(len(other_indices))[:num_negatives]]
            else:
                sampled_idx = torch.randint(0, N_masked, (num_negatives,))
            negs.append(q_norm[sampled_idx])
            
        negs = torch.stack(negs)  # (N_masked, num_negatives=10, 512)
        
        pos_sim = torch.sum(c_norm * q_norm, dim=-1, keepdim=True) / temp
        neg_sim = torch.bmm(negs, c_norm.unsqueeze(-1)).squeeze(-1) / temp
        
        logits = torch.cat([pos_sim, neg_sim], dim=-1) # (N_masked, 1 + 10 = 11)
        labels = torch.zeros(N_masked, dtype=torch.long, device=c.device)
        return F.cross_entropy(logits, labels)

    def forward(self, raw_audio):
        z = self.feature_extractor(raw_audio)
        q_targets, prob_perplexity, avg_probs = self.quantizer(z)
        q_projected = self.project_q(q_targets)
        z_masked, mask_indices = self.apply_masking(z)
        c = self.transformer(z_masked)
        
        contrastive_loss = self.compute_contrastive_loss(c, q_projected, mask_indices)
        diversity_loss = (avg_probs * torch.log(avg_probs + 1e-7)).sum(dim=-1).mean()
        total_loss = contrastive_loss + 0.1 * diversity_loss
        
        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "diversity_loss": diversity_loss,
            "z_shape": z.shape,
            "c_shape": c.shape,
            "q_shape": q_targets.shape,
            "masked_steps": mask_indices.sum().item(),
            "perplexity": prob_perplexity.item()
        }
```

---

# 7. Pre-Training Loss Functions & Worked Numerical Example

### Combined Pre-Training Loss
$$\mathcal{L} = \mathcal{L}_m + \alpha \mathcal{L}_d \quad (\text{where } \alpha = 0.1)$$

### Contrastive Loss ($\mathcal{L}_m$)
$$\mathcal{L}_m = -\log \frac{\exp( \text{sim}(c_t, q_t) / \kappa )}{\sum_{\tilde{q} \in \mathbf{Q}_t} \exp( \text{sim}(c_t, \tilde{q}) / \kappa )}$$
where $\text{sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$ (Cosine Similarity) and temperature $\kappa = 0.1$.

### Worked Step-by-Step Loss Calculation
Given temperature $\kappa = 0.1$ and $K=2$ distractors:
- **True Target Similarity:** $\text{sim}(c_t, q_t) = 0.80 \implies \text{Score} = 0.80 / 0.1 = 8.0 \implies \exp(8.0) \approx 2980.958$
- **Distractor 1 Similarity:** $\text{sim}(c_t, \tilde{q}_1) = 0.20 \implies \text{Score} = 0.20 / 0.1 = 2.0 \implies \exp(2.0) \approx 7.389$
- **Distractor 2 Similarity:** $\text{sim}(c_t, \tilde{q}_2) = -0.10 \implies \text{Score} = -0.10 / 0.1 = -1.0 \implies \exp(-1.0) \approx 0.368$

- **Softmax Probability for True Target:**
  $$P(\text{True Target}) = \frac{2980.958}{2980.958 + 7.389 + 0.368} = \frac{2980.958}{2988.715} \approx \mathbf{0.9974} \text{ (99.74\%)}$$

- **Contrastive Loss:** $\mathcal{L}_m = -\log(0.9974) \approx \mathbf{0.0026}$

---

# 8. Experimental Results & Benchmarks

| Metric / Setting | Baseline / Previous SOTA | wav2vec 2.0 (BASE) | wav2vec 2.0 (LARGE) |
| :--- | :--- | :--- | :--- |
| **Unlabeled Pre-training Data** | N/A | 960h (LibriSpeech) | 53k hours (LibriVox) |
| **WER (10 Minutes Labeled Data)** | 25.2% (Discrete BERT) | 9.1% (Clean) / 15.6% (Other) | **4.8% (Clean) / 8.2% (Other)** |
| **WER (1 Hour Labeled Data)** | 9.0% (Discrete BERT) | 5.5% (Clean) / 11.3% (Other) | **2.9% (Clean) / 5.8% (Other)** |
| **WER (Full 960h Labeled Data)** | 2.1% (ContextNet) | 2.1% (Clean) / 4.8% (Other) | **1.8% (Clean) / 3.3% (Other)** |

---

# 9. Practical Fine-Tuning Guide & Real-World Use Cases

```text
                        ┌──► 1. Speech-to-Text (ASR) [Transcribe custom domain speech]
                        │
 Pre-trained wav2vec2 ──┼──► 2. Emotion & Sentiment Recognition [Call centers, therapy AI]
 (Hugging Face / PyTorch)│
                        ├──► 3. Keyword Spotting & Voice Commands ["Hey Siri", IoT controls]
                        │
                        └──► 4. Speaker Identification & Verification [Voice security]
```

### Task Recipe Table

| Task | Output Head | Pooling Strategy | Training Loss |
| :--- | :--- | :--- | :--- |
| **Speech-to-Text (ASR)** | Linear projection to character vocab | None (Sequence Output) | **CTC Loss (`nn.CTCLoss`)** |
| **Emotion Recognition** | Linear head to $N$ emotion classes | **Mean Pooling** over time | **Cross-Entropy (`nn.CrossEntropyLoss`)** |
| **Keyword / Voice Command** | Linear head to $K$ keyword classes | **Mean / Max Pooling** over time | **Cross-Entropy (`nn.CrossEntropyLoss`)** |
| **Speaker Verification** | Cosine Metric Embedding Head | **Mean Pooling** over time | **Triplet / Contrastive Loss** |
