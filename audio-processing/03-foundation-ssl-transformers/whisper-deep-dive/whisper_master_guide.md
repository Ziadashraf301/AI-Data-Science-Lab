# Whisper: Technical Guide

- **Paper:** *Robust Speech Recognition via Large-Scale Weak Supervision*, Radford, Kim, Xu, Brockman, McLeavey, Sutskever (OpenAI, 2022)
- **arXiv:** [2212.04356](https://arxiv.org/abs/2212.04356)
- **Code:** [github.com/openai/whisper](https://github.com/openai/whisper)
- **Notebook:** [whisper_implementation.ipynb](whisper_implementation.ipynb) (all numbers in section 9 come from it)

## Contents
1. [Overview](#1-overview)
2. [Input: log-Mel spectrogram](#2-input-log-mel-spectrogram)
3. [Audio encoder](#3-audio-encoder)
4. [Text decoder](#4-text-decoder)
5. [Multitask token format](#5-multitask-token-format)
6. [Long-form decoding](#6-long-form-decoding)
7. [Arabic and English](#7-arabic-and-english)
8. [Deployment: offline vs online](#8-deployment-offline-vs-online)
9. [Results on my data](#9-results-on-my-data)
10. [Summary](#10-summary)

---

## 1. Overview

Whisper is a standard Transformer encoder-decoder. The architecture is not new; what's different is the training data: 680,000 hours of audio with transcripts scraped from the web. The labels are noisy, so the paper calls it weak supervision. Because the data is so large and varied, the model works zero-shot on new audio without fine-tuning.

| | wav2vec 2.0 / HuBERT / WavLM | Whisper |
|---|---|---|
| Training | Self-supervised (audio only) | Weakly supervised (audio + web transcripts) |
| Architecture | Encoder only + CTC head | Encoder + decoder (seq2seq) |
| Input | Raw waveform (CNN feature encoder) | Log-Mel spectrogram |
| ASR out of the box | No, needs fine-tuning on labelled data | Yes |
| Output | Characters (CTC) | BPE tokens, with punctuation and casing |

Training data split (from the paper): about 438k hours of English ASR, 117k hours of non-English ASR across 96 languages, and 125k hours of X to English translation.

---

## 2. Input: log-Mel spectrogram

| Step | Value | Shape (30 s) |
|---|---|---|
| Resample | 16 kHz mono | 480,000 samples |
| Pad / trim | always exactly 30 s | 480,000 |
| STFT | window 400 (25 ms), hop 160 (10 ms) | 3,000 frames |
| Mel filterbank | 80 bins (128 in large-v3) | 80 x 3,000 |
| Log + normalise | `log10`, clamp to `max - 8`, then `(x + 4) / 4` | values around [-1, 1] |

A 7 second clip is 112,000 samples, zero-padded to 480,000. So most of the 80 x 3,000 matrix is silence. This padding is one reason Whisper can hallucinate text on short clips.

---

## 3. Audio encoder

```text
log-Mel (80 x 3000)
  -> Conv1D(80 -> d, k=3, pad=1) + GELU            d x 3000
  -> Conv1D(d -> d, k=3, stride=2, pad=1) + GELU   d x 1500   (halves time)
  -> + sinusoidal positional encoding
  -> N x [LayerNorm -> Self-Attention -> LayerNorm -> MLP]   (pre-LN, residual)
  -> LayerNorm -> H_audio (1500 x d)
```

- Each of the 1,500 output vectors covers 20 ms of audio.
- Self-attention is O(1500^2), so the stride-2 conv halves the attention cost.

| Model | Layers (enc / dec) | d_model | Heads | Params | VRAM fp16 (approx.) |
|---|---|---|---|---|---|
| tiny | 4 / 4 | 384 | 6 | 39 M | ~1 GB |
| base | 6 / 6 | 512 | 8 | 74 M | ~1 GB |
| small | 12 / 12 | 768 | 12 | 244 M | ~2 GB |
| medium | 24 / 24 | 1024 | 16 | 769 M | ~5 GB |
| large-v2 / v3 | 32 / 32 | 1280 | 20 | 1.55 B | ~10 GB |
| large-v3-turbo | 32 / 4 | 1280 | 20 | 809 M | ~6 GB |

`.en` variants (tiny.en to medium.en) are English-only and slightly better on English. Turbo keeps the full large-v3 encoder but cuts the decoder from 32 layers to 4, and decoding is where most of the time goes.

---

## 4. Text decoder

```text
tokens -> token embedding + learned positions (max 448)
  -> N x [Masked Self-Attn -> Cross-Attn(Q = text, K,V = H_audio) -> MLP]
  -> LayerNorm -> tied output embedding -> softmax over 51,865 tokens
```

- Cross-attention is how each text position looks at the 1,500 audio vectors.
- Tokenizer: byte-level BPE (GPT-2 style, extended for other scripts). Arabic takes more tokens per word than English, so decoding is slower and the 448-token context fills faster.
- Loss: next-token cross-entropy on the text tokens only, not on the prompt tokens.

Shapes I traced with forward hooks on `base` (d = 512): conv1 `(1, 512, 3000)`, conv2 `(1, 512, 1500)`, encoder `(1, 1500, 512)`, logits `(1, 4, 51865)`.

---

## 5. Multitask token format

The decoder is told what to do through a prefix of special tokens:

```text
<|startoftranscript|> <|ar|> <|transcribe|> <|notimestamps|> ...text... <|endoftext|>
       start          language    task       timestamp mode              stop
```

| Goal | Prefix |
|---|---|
| Arabic speech to Arabic text | `<\|startoftranscript\|><\|ar\|><\|transcribe\|>` |
| Arabic speech to English text | `<\|startoftranscript\|><\|ar\|><\|translate\|>` |
| Language detection | feed only `<\|startoftranscript\|>`, read the language-token probabilities |
| Voice activity | probability of `<\|nospeech\|>` |
| Timestamps | leave out `<\|notimestamps\|>`; the model emits time tokens every 20 ms |
| Prompting (names, spelling) | previous text after `<\|startofprev\|>` |

Whisper only translates into English. English to Arabic needs a separate MT model or an LLM.

---

## 6. Long-form decoding

The model sees 30 s at a time, so `transcribe()` runs a sliding window:

1. Decode a window. The last predicted timestamp sets where the next window starts.
2. Condition on the previous window's text to keep names and spelling consistent.
3. Temperature fallback: start at T = 0 (greedy or beam 5). If the output is bad, retry at T = 0.2, 0.4, ..., 1.0. Bad means:
   - gzip compression ratio > 2.4 (repetition loops), or
   - average log-prob < -1.
4. No-speech: if P(`<|nospeech|>`) > 0.6 and the log-prob is low, skip the window.

These checks are the main protection against hallucination (fluent made-up text during silence or music). In practice: run VAD before decoding, set `condition_on_previous_text=False` if loops show up, and avoid long silences.

Two things to know:
- Language detection runs once, on the first 30 s, and is then fixed for the whole file. Code-switched audio needs to be split, or the language set per chunk.
- `avg_logprob`, `compression_ratio` and `no_speech_prob` are computed per 30 s window, not per segment. Every segment in a window gets the same values.

---

## 7. Arabic and English

What works:
- One model for both languages, including English words inside Arabic sentences.
- Punctuated, readable output without a separate punctuation model.
- Zero-shot: no Arabic labelled data needed to start.
- Translation to English with `task="translate"` (large models only, see section 9).

Limitations for Arabic:

| Issue | Cause | Fix |
|---|---|---|
| Much lower accuracy than English | Small share of the training data | Use large-v3 / turbo, or fine-tune |
| Dialects (Egyptian, Gulf, Levantine) | Data skews to MSA and news-style speech | Fine-tune on dialect data (Common Voice, MGB-3, own data) |
| Dialect phonetics | Egyptian ج is pronounced g, ث often as س | Fine-tuning, or lexicon correction |
| No diacritics | Web transcripts rarely have them | Post-processing diacritisation model |
| Spelling variants (ة/ه, أ/ا, ى/ي) | Noisy web labels | Normalise before WER |
| English loanwords | Model writes them in Latin script, references often in Arabic script | Normalise or transliterate before scoring |
| Language mis-detection on short clips | Too little audio to decide | Always pass `language="ar"` |

Evaluation: report WER and CER together for Arabic. One wrong letter makes the whole word wrong, so WER alone overstates the error.

Normalisation used in the notebook: remove tashkeel and tatweel, unify ا/أ/إ/آ, ة to ه, ى to ي, remove punctuation.

---

## 8. Deployment: offline vs online

| | Offline | Online (API) |
|---|---|---|
| Tools | `openai-whisper`, faster-whisper (CTranslate2), whisper.cpp, HF `transformers` | Groq, Hugging Face Inference, OpenAI (`whisper-1`, `gpt-4o-transcribe`), Azure |
| Privacy | Audio stays on the machine | Audio sent to a third party |
| Cost | Hardware only | Per minute of audio (after free tiers) |
| Internet | Only to download weights once | Every request |
| Control | Fine-tuning, custom decoding settings | Fixed models, limited parameters |
| Limits | You manage the hardware | Request size limits (e.g. 25 MB on OpenAI) |

Running fully offline with faster-whisper: download the model once, then load with `local_files_only=True` and `HF_HUB_OFFLINE=1`. To deploy on a machine with no internet, copy `~/.cache/huggingface` over.

Notes from the notebook:
- faster-whisper needs cuBLAS 12 and cuDNN 9 on GPU. If Colab's torch ships other versions, install `nvidia-cublas-cu12` and `nvidia-cudnn-cu12==9.*` and preload them.
- Newer PyAV versions break faster-whisper's file decoding (`metadata_errors` error). Pass a decoded 16 kHz numpy array instead of a path.

Choosing:
- Prototype or low volume, non-sensitive audio: online API.
- Private data, no internet, high volume, or dialect fine-tuning: faster-whisper on GPU, or whisper.cpp on CPU / edge.
- Live captions: streaming wrappers (`whisper_streaming`, `WhisperLive`) with VAD and short chunks.

---

## 9. Results on my data

Test data: a 69.5 s Egyptian Arabic clip (with English product words) and a 5.9 s English LibriSpeech sample. Run on Colab with a T4 GPU.

### Model size (openai-whisper, GPU, greedy)

| Model | EN WER | AR WER | AR CER | AR speed (x real time) |
|---|---|---|---|---|
| tiny | 0.059 | 0.88 | 0.49 | 8.1 |
| base | 0.059 | 0.76 | 0.27 | 14.8 |
| small | 0.059 | 0.45 | 0.13 | 7.9 |
| turbo | 0.059 | 0.24 | 0.11 | 14.6 |
| tiny.en / base.en | 0.059 | - | - | - |

### CPU vs GPU (faster-whisper, beam 5, VAD, Arabic clip)

| Model | CPU int8 (x RT) | GPU fp16 (x RT) | WER CPU / GPU |
|---|---|---|---|
| tiny | 5.6 | 27.4 | 0.84 / 0.83 |
| base | 3.6 | 12.2 | 0.69 / 0.76 |
| small | 1.2 | 19.4 | 0.47 / 0.43 |
| large-v3-turbo | - | 19.4 | - / 0.21 |

### Online APIs (Arabic clip)

| Provider / model | Time | WER |
|---|---|---|
| Groq whisper-large-v3 | 1.8 s | 0.21 |
| Groq whisper-large-v3-turbo | 0.8 s | 0.24 |

### Findings
- English is at the ceiling for every model (1 wrong word out of 17), so this sample can't separate them.
- Arabic WER drops from 0.88 to 0.21 with model size. Size matters much more for Arabic than for English.
- Turbo decodes as fast as base on GPU while being about 3x more accurate, because it only has 4 decoder layers.
- int8 on CPU loses almost nothing in accuracy compared to fp16 on GPU.
- faster-whisper with beam 5 + VAD (0.21) beat openai-whisper greedy (0.24) on the same turbo weights.
- The APIs give the same WER as the local runs because they serve the same open weights. Online vs offline is a choice about privacy, cost and hardware, not accuracy.
- `translate` with `base` produced unrelated hallucinated text. Dialect-to-English translation needs large or turbo.
- Most of the remaining ~0.21 WER is dialect spelling (الحجز decoded as الحقز) and English words in Latin script, not misheard audio.

---

## 10. Summary

```text
audio -> log-Mel (80 x 3000) -> Encoder -> H_audio (1500 x d)
                                               | cross-attention (K, V)
prefix tokens ------------------------------> Decoder -> text, timestamps, language
```

1. Fixed 30 s window: 80 x 3000 Mel, 1500 encoder vectors.
2. Standard encoder-decoder; the decoder reads the audio through cross-attention.
3. Special tokens set the language, the task and timestamps.
4. Long audio uses a sliding window, temperature fallback and no-speech checks.
5. For Arabic: use turbo or large-v3, set `language="ar"`, report CER with WER, and fine-tune for dialects.
