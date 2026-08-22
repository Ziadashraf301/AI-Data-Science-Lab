# 🎙️ Speech AI Roadmap & Milestones (2015 – 2026)

This roadmap outlines the most critical concepts, architectural breakthroughs, and modeling paradigm shifts in Speech AI (Automatic Speech Recognition, Text-to-Speech, Speech Translation, and Speech Representation) from 2015 to 2026.

---

## 📅 2015 – 2017: The Deep RNN & Autoregressive Waveform Era
The transition from traditional statistical models (HMM-GMM) to end-to-end Deep Neural Networks.

*   **Key Concepts**:
    *   **CTC Loss (Connectionist Temporal Classification)**: Allowed alignment-free training of speech-to-text models.
    *   **Seq2Seq with Attention**: Used encoder-decoder models to directly translate spectrogram frames to characters.
*   **Architectures to Know**:
    *   **DeepSpeech 1 & 2 (Baidu)**: Large RNN/GRU architectures trained end-to-end using CTC loss.
    *   **WaveNet (DeepMind)**: Autoregressive CNN model generating raw audio waveforms pixel-by-pixel. Set the gold standard for speech naturalness but was extremely slow.
    *   **Tacotron (Google)**: End-to-end Text-to-Spectrogram model using sequence-to-sequence attention.

---

## 📅 2018 – 2020: The Rise of Transformers & Conformers
Speech models transitioned to attention mechanisms, enabling parallel training and modeling of long-range acoustic structures.

*   **Key Concepts**:
    *   **Neural Vocoders**: Models that quickly convert 2D mel-spectrograms back into 1D waveforms (replacing autoregressive models like WaveNet).
    *   **Self-Attention in Audio**: Using self-attention to capture global context across long audio time frames.
*   **Architectures to Know**:
    *   **Conformer (Google)**: Fused CNNs (for local feature extraction) with Transformers (for global context). Still one of the most dominant backbones for ASR today.
    *   **WaveGlow / MelGAN**: Non-autoregressive GAN-based/Flow-based vocoders, making TTS synthesis real-time.

---

## 📅 2020 – 2022: Self-Supervised Learning (SSL) & Foundation Models
The biggest shift in Speech AI: pre-training on thousands of hours of unlabeled audio to learn speech representations.

*   **Key Concepts**:
    *   **Contrastive Loss in Speech**: Learning representations by predicting masked audio segments from distractors.
    *   **Speech Tokenization**: Discretizing raw audio waves into codebook vectors.
*   **Architectures to Know**:
    *   **Wav2Vec 2.0 (Meta)**: Learns representations of raw speech by solving a contrastive task over masked latent representations.
    *   **HuBERT & WavLM (Meta/Microsoft)**: Uses k-means clustering to generate pseudo-labels for speech, training the model like BERT. WavLM added speech denoising tasks.
    *   **AST (Audio Spectrogram Transformer)**: Self-attention applied directly to 2D spectrogram patches, proving ViT works for audio.

---

## 📅 2023 – 2024: Zero-Shot TTS & Robust Open-Source ASR
Paradigm shifts in robust multilingual speech recognition and zero-shot voice cloning.

*   **Key Concepts**:
    *   **Neural Audio Codecs**: Quantizing audio into discrete tokens (e.g., EnCodec) to treat audio generation as a language modeling problem.
    *   **Zero-Shot TTS**: Cloning any voice using just a 3-second reference prompt.
*   **Architectures to Know**:
    *   **Whisper (OpenAI)**: Encoder-decoder Transformer trained supervised on 680,000 hours of multilingual, multi-task data. Set the standard for zero-shot, robust ASR.
    *   **VALL-E (Microsoft)**: Neural codec language model for zero-shot TTS. Treated voice synthesis as a next-token prediction task.

---

## 📅 2025 – 2026: Native Multimodal LLMs & Real-Time Duplex Audio
The current SOTA: models that understand and generate speech natively, without intermediate text representations.

*   **Key Concepts**:
    *   **Native Multimodal Tokenization**: Feeding speech tokens directly into LLMs alongside text/image tokens.
    *   **Duplex Streaming Speech**: Real-time voice agents that listen, speak, handle interruptions, and express emotion (prosody, laughter) dynamically.
*   **Architectures to Know**:
    *   **GPT-4o / Gemini 1.5 (OpenAI/Google)**: End-to-end multimodal models. Audio input is directly tokenized and processed by the LLM core, reducing latency and retaining emotional tone.
    *   **Speech-to-Speech (S2S) Models (e.g., AudioPaLM)**: Translates audio in one language directly to audio in another, retaining the speaker's original voice characteristics.
