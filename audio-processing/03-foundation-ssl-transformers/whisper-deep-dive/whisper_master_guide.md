# 📖 Whisper: The Minute Master Guide

> **Paper:** *Robust Speech Recognition via Large-Scale Weak Supervision*
> **Authors:** Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever (OpenAI, 2022)
> **arXiv:** [2212.04356](https://arxiv.org/abs/2212.04356) | **Code:** [github.com/openai/whisper](https://github.com/openai/whisper)
> **Companion notebook:** [whisper_implementation.ipynb](whisper_implementation.ipynb)

## ⏱️ 30-Minute Reading Plan

| Minutes | Section                                   | You will be able to...               |
| ------- | ----------------------------------------- | ------------------------------------ |
| 0–4    | 1. Big idea                               | explain Whisper in one sentence      |
| 4–12   | 2–4. Input → Encoder → Decoder         | trace shapes from raw audio to text  |
| 12–16  | 5. Multitask tokens                       | read and write Whisper's task prompt |
| 16–20  | 6. Long-form decoding                     | explain hallucination guards         |
| 20–26  | 7–8. Arabic & English, offline vs online | choose a deployment                  |
| 26–30  | 9–10. Comparison, recap & Arabic post    | write the LinkedIn post              |

---

## 1. The Big Idea (in one sentence)

> **Whisper doesn't use a new architecture. It uses a *standard* Transformer encoder–decoder trained on a *huge, noisy, multilingual* dataset (680,000 hours scraped from the web), so it works zero-shot on audio it has never seen.**

How it differs from the models in this folder:

|              | wav2vec 2.0 / HuBERT / WavLM                    | Whisper                                                     |
| ------------ | ----------------------------------------------- | ----------------------------------------------------------- |
| Learning     | **Self-supervised** (audio only, no text) | **Weakly supervised** (audio + noisy web transcripts) |
| Architecture | Encoder only + CTC head                         | **Encoder + Decoder** (seq2seq)                       |
| Input        | Raw waveform (CNN feature encoder)              | **Log-Mel spectrogram**                               |
| Use as ASR   | Needs fine-tuning on labelled data              | **Works out of the box**                              |
| Output       | Characters (CTC)                                | BPE text tokens, with punctuation and casing                |

**Analogy:** wav2vec2/HuBERT learn to *hear* (good acoustic features). Whisper learns to *listen and write*, like a stenographer who has transcribed 77 years of audio.

---

## 2. Step 1: Audio → Log-Mel Spectrogram

| Step            | Value                                                | Shape (30 s example)                     |
| --------------- | ---------------------------------------------------- | ---------------------------------------- |
| Resample        | 16,000 Hz mono                                       | 30 × 16,000 = **480,000 samples** |
| Pad / trim      | Always exactly **30 s**                        | 480,000                                  |
| STFT            | window 400 (25 ms), hop 160 (10 ms)                  | 480,000 / 160 = **3,000 frames**   |
| Mel filterbank  | **80** bins (128 in large-v3)                  | **80 × 3,000**                    |
| Log + normalise | `log10`, clamp to (max − 8), then `(x + 4) / 4` | values ≈ [-1, 1]                        |

**Worked example:** a 7-second voice note becomes 112,000 samples, is padded with zeros up to 480,000, and becomes an 80 × 3,000 matrix. The last ~2,300 columns are just silence. Whisper *always* sees 30 s, which is why short clips can sometimes "hallucinate" text in the silent part.

---

## 3. Step 2: The Audio Encoder

```text
log-Mel (80 × 3000)
  → Conv1D(80→d, k=3, pad=1) + GELU          → d × 3000
  → Conv1D(d→d,  k=3, stride=2, pad=1) + GELU → d × 1500   ← halves time
  → + sinusoidal positional encoding
  → N × [LayerNorm → Self-Attention → LayerNorm → MLP] (pre-LN, residual)
  → LayerNorm → H_audio ∈ ℝ^(1500 × d)
```

- Each of the 1,500 output vectors covers **20 ms** of audio.
- Self-attention cost grows with 1500², so the conv stride matters.

| Model          | Layers (enc = dec)       | d_model | Heads | Params | VRAM (fp16, approx.) | Relative speed |
| -------------- | ------------------------ | ------- | ----- | ------ | -------------------- | -------------- |
| tiny           | 4                        | 384     | 6     | 39 M   | ~1 GB                | ~10×          |
| base           | 6                        | 512     | 8     | 74 M   | ~1 GB                | ~7×           |
| small          | 12                       | 768     | 12    | 244 M  | ~2 GB                | ~4×           |
| medium         | 24                       | 1024    | 16    | 769 M  | ~5 GB                | ~2×           |
| large (v2/v3)  | 32                       | 1280    | 20    | 1.55 B | ~10 GB               | 1×            |
| large-v3-turbo | 32 enc / **4 dec** | 1280    | 20    | 809 M  | ~6 GB                | ~8×           |

`.en` versions (tiny.en … medium.en) are English-only and slightly better for English.

---

## 4. Step 3: The Text Decoder

```text
tokens so far → token embedding + learned positions (max 448)
  → N × [Masked Self-Attn → Cross-Attn(Q=text, K,V=H_audio) → MLP]
  → LayerNorm → (tied) output embedding → softmax over ~51,865 tokens
```

- **Cross-attention** is the "listening" step. Each text position looks at the 1,500 audio frames and decides which ones to attend to.
- Tokenizer: byte-level BPE (GPT-2 style; the multilingual version adds tokens for other scripts).
  ⚠️ Arabic text needs **more tokens per word** than English, so Arabic decoding is slower and the 448-token limit fills faster.
- Training loss: standard next-token cross-entropy, computed only on the text tokens (not on the prompt).

---

## 5. Multitask Format: One Model, Many Jobs

The decoder is "told" what to do by a token prefix:

```text
<|startoftranscript|> <|ar|> <|transcribe|> <|notimestamps|>  مرحبا بكم ...  <|endoftext|>
        start           language     task         timestamp mode     text           stop
```

| Goal                                    | Prefix                     |
| --------------------------------------- | -------------------------- |
| Arabic speech → Arabic text            | `<                         |
| Arabic speech → **English** text | `<                         |
| Detect language                         | feed only `<               |
| Detect silence                          | probability of `<          |
| Timestamps                              | drop `<                    |
| Context / spelling hints                | put previous text after `< |

> Whisper only translates **into English**. For English → Arabic you need a separate MT model or an LLM.

---

## 6. Long-form Decoding (audio longer than 30 s)

The model only sees 30 s at a time, so `transcribe()` uses a sliding window:

1. Decode a window. Use the **last predicted timestamp** to decide where the next window starts.
2. **Condition** on the previous window's text, for consistency of names and spelling.
3. **Temperature fallback:** start greedy/beam (T=0, beam 5). If the output looks broken, retry with T = 0.2, 0.4 … 1.0. "Broken" means either:
   - gzip **compression ratio > 2.4** (repetitive loops like "شكرا شكرا شكرا..."), or
   - **average log-prob < −1** (low confidence).
4. **No-speech:** if P(`<|nospeech|>`) > 0.6 *and* confidence is low, skip the window.

These heuristics are the main defence against **hallucination**, Whisper's best-known weakness: it can invent fluent text during silence or music. Practical fixes: use VAD (voice activity detection) before transcribing, set `condition_on_previous_text=False` if loops appear, and avoid long silences.

---

## 7. Whisper for Arabic 🇪🇬 and English 🇬🇧

### Why Whisper is a strong choice

- **One model handles both** languages and **code-switching** (Egyptian speakers mixing English tech words into Arabic sentences).
- Outputs **punctuation and readable text** directly, with no separate punctuation model.
- **Zero-shot:** no Arabic labelled data needed to get started.
- **Translation to English** for free (`task="translate"`).

### The honest limitations for Arabic

| Issue                                       | Why                                                                             | Fix                                                                          |
| ------------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Much lower accuracy than English            | Arabic had only a small share of the 680k hours (English ASR alone was ~438k h) | Use `large-v3` / `turbo`, or fine-tune                                   |
| Dialects (Egyptian, Gulf, Levantine)        | Training data skews to MSA (Modern Standard Arabic) and news-style speech       | Fine-tune on dialect data (e.g. Common Voice, MGB-3 Egyptian, your own data) |
| No diacritics (تشكيل)                  | Web transcripts rarely have them                                                | A post-processing diacritization model                                       |
| Inconsistent spelling (ة/ه, أ/ا, ى/ي) | Noisy web labels                                                                | Normalize before computing WER                                               |
| Language mis-detection on short clips       | Very little audio to decide from                                                | Always pass `language="ar"`                                                |

**English** works well out of the box. For English-only, `distil-whisper` and `.en` models are faster.

> 📏 **Measure, don't guess:** compute **WER** (word error rate) and, for Arabic, **CER** (character error rate) on 50–100 of *your own* clips. The notebook shows how.

---

## 8. Offline vs Online: How to Deploy

|          | **Offline (on your machine)**                                                                                                         | **Online (cloud API)**                                                           |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Tools    | `openai-whisper`, **faster-whisper** (CTranslate2, int8), **whisper.cpp** (CPU / Apple Silicon / mobile), HF `transformers` | OpenAI API (`whisper-1`, `gpt-4o-transcribe`), Groq, Hugging Face Inference, Azure |
| Privacy  | ✅ Audio never leaves the device (medical, legal, call centres)                                                                             | ⚠️ Audio is sent to a third party                                                    |
| Cost     | Hardware only, no per-minute fee                                                                                                            | Pay per minute of audio                                                                |
| Internet | Only once, to download weights (tiny 75 MB … large ~3 GB)                                                                                  | Every request                                                                          |
| Speed    | Depends on GPU/CPU; faster-whisper int8 runs `small` in real time on CPU                                                                  | Fast, scales automatically                                                             |
| Control  | Fine-tune for Egyptian Arabic, custom vocabulary                                                                                            | Fixed models, limited parameters                                                       |
| Limits   | You manage hardware                                                                                                                         | File size limits (e.g. 25 MB per request on OpenAI)                                    |

**Rule of thumb**

- Prototype, or small volume without sensitive data → **online API**.
- Private data, offline sites, high volume, or dialect fine-tuning → **offline faster-whisper** (GPU) or **whisper.cpp** (CPU/edge).
- Live captions → streaming wrappers (e.g. `whisper_streaming`, `WhisperLive`) with VAD and short chunks.

---

## 9. Quick Comparison & Mental Model

```text
          ┌──────────── Whisper ────────────┐
audio ──► log-Mel ──► Encoder ──► H_audio ──┐
                                            ▼ (cross-attn K,V)
prefix tokens ─────────────────────► Decoder ──► text, timestamps, language
```

**5 things to remember**

1. 30 s window → 80 × 3000 Mel → 1500 audio vectors.
2. Encoder–decoder Transformer; the decoder "listens" through cross-attention.
3. Special tokens choose the language, the task and timestamps.
4. Long audio is handled by a sliding window, temperature fallback and no-speech checks to limit hallucination.
5. Arabic works but needs `large-v3`/`turbo`, `language="ar"` and ideally dialect fine-tuning. Run offline for privacy, online for convenience.

---

## 10. ✍️ Ready-to-adapt LinkedIn Post (Arabic)

> **Whisper: إزاي موديل واحد بيسمع ويكتب ١٠٠ لغة؟ 🎙️**
>
> Whisper من OpenAI مش معماريّة جديدة… السرّ في **الداتا**: ٦٨٠ ألف ساعة صوت من الإنترنت بنصوصها (حتى لو فيها أخطاء).
>
> 🔹 الصوت بيتقطع لـ ٣٠ ثانية ويتحول لـ Log-Mel Spectrogram (80×3000)
> 🔹 الـ Encoder بيحوله لـ 1500 vector، والـ Decoder "بيسمع" عن طريق Cross-Attention
> 🔹 Tokens خاصة بتحدد المهمة: `<|ar|><|transcribe|>` أو `<|translate|>` للإنجليزي
>
> **ليه مفيد لينا كعرب؟**
> ✅ عربي + إنجليزي في نفس الجملة (code-switching)
> ✅ يشتغل **Offline** على جهازك (faster-whisper / whisper.cpp) → خصوصية كاملة
> ✅ أو **Online** عبر API لو عايز سرعة من غير GPU
> ⚠️ اللهجة المصري محتاجة Fine-tuning عشان دقة أعلى
>
> 📄 الورقة: Robust Speech Recognition via Large-Scale Weak Supervision (arXiv:2212.04356)
>
> #AI #SpeechRecognition #Whisper #Arabic_NLP

*(Pair it with the cheat sheet image in `brand/whisper_cheatsheet.png`.)*
