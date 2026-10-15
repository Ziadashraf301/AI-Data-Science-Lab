# NLP / NLU

Text classification experiments.

```text
nlp/
└── safety-classifiers-benchmark/
    └── benchmark_jev_vs_bert_vs_gemini.ipynb   # Spam/phishing + jailbreak detection: Jev vs DeBERTa (zero-shot) vs Gemini Flash-Lite
```

## Safety classifiers benchmark
I compare three different kinds of models on two safety tasks, spam/phishing detection and jailbreak / prompt injection detection:
- **Jev** (TypeSafe API), a single-pass decision model
- **DeBERTa** (`cross-encoder/nli-deberta-v3-xsmall`), a small local zero-shot classifier
- **Gemini Flash-Lite**, an LLM called in batches

I look at accuracy, F1, calibration (Brier score) and latency.

API keys are read from Colab secrets or environment variables (`TYPESAFE_API_KEY`, `GEMINI_API_KEY`). Don't put keys in the notebook.
