# Advanced Tree-Based & Tabular Machine Learning Laboratory (Classic to 2026)
## Enterprise E-Commerce 20-Class Benchmark (3M Rows, 2,000 Features, Bilingual AR/EN)

---

## 1. Executive Summary

This repository is an end-to-end production laboratory and educational curriculum dedicated to **Tree-Based Machine Learning and Tabular Architectures**. It covers the complete evolution of tabular methods—from **CART (1980s)** and **Random Forests (2001)** to the **"Big Three" GBDTs**, **Glass-Box Interpretable Models**, **Streaming Real-Time Trees**, **Neural-Tree Hybrids (2020–2024)**, and **Tabular Foundation Models (TFMs - 2022–2026)**.

The entire suite is benchmarked on a realistic **20-Class E-Commerce Next-Purchase Department Prediction Problem** featuring bilingual (**Arabic & English**) categorical and continuous transactional attributes across up to **3 Million rows and 2,000 tabular features**.

---

## 2. The 20-Class Bilingual E-Commerce Classification Problem

### The Business Objective
In regional e-commerce platforms (such as **Noon, Amazon MENA, Jumia, and Jarir**), real-time recommendation engines and search re-rankers predict:
> **"Which of the 20 product departments is this customer most likely to purchase from during this session?"**

### The 20 Target Departments
1. **Mobiles & Tablets** (هواتف وأجهزة لوحية)
2. **Consumer Electronics** (إلكترونيات استهلاكية)
3. **Computers & Gaming** (حواسيب وألعاب فيديو)
4. **Home & Kitchen** (المنزل والمطبخ)
5. **Large Appliances** (أجهزة منزلية كبرى)
6. **Men's Fashion** (أزياء رجالية)
7. **Women's Fashion** (أزياء نسائية)
8. **Baby & Kids** (أطفال ومواليد)
9. **Beauty & Fragrances** (عطور ومستحضرات تجميل)
10. **Personal Care & Health** (عناية شخصية وصحة)
11. **Supermarket & Groceries** (سوبرماركت وبقالة)
12. **Sports & Fitness** (رياضة ولياقة بدنية)
13. **Automotive & Hardware** (سيارات ومعدات صيانة)
14. **Books & Stationery** (كتب ومستلزمات مكتبية)
15. **Toys & Games** (ألعاب ترفيهية)
16. **Furniture & Decor** (أثاث وديكور منزلي)
17. **Watches & Jewelry** (ساعات ومجوهرات)
18. **Pet Supplies** (مستلزمات حيوانات أليفة)
19. **Office Electronics** (تجهيزات مكتبية)
20. **Travel & Luggage** (حقائب ومستلزمات سفر)

### Tabular Input Attributes
* **Arabic Categorical Signals**: Governorate & City (`القاهرة`, `الرياض`, `الإسكندرية`, `جدة`...), Payment Method (`الدفع عند الاستلام`, `بطاقة مدى`, `تابي/تمارا`, `تحويل بنكي`), Customer Loyalty Tier (`عميل VIP`, `عميل جديد`...), Shipping Status (`تم التوصيل`, `قيد الشحن`...).
* **English Categorical Signals**: Device OS (`iOS App`, `Android App`, `Desktop Web`), Acquisition Channel (`Google Ads`, `Organic Search`, `TikTok Ads`), Courier Partner (`Aramex`, `DHL Express`, `SMSA`), Fulfillment Hub (`HUB-Cairo-East`, `HUB-Riyadh-North`).
* **Continuous & Derived Interaction Features**: Basket value, discount percentage, session duration, clicks, return rates, RFM recency/frequency/monetary scores, value-per-item, click velocities, and interaction aggregates expanding up to **2,000 tabular features**.

---

## 3. Sequential 7-Notebook Curriculum

The notebooks are organized progressively inside [trees/notebooks/](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/trees/notebooks):

```
trees/
├── data_generator.py                                                # Bilingual 3M-row, 2000-feature Parquet generator (~500 MB)
├── utils.py                                                         # Unified multi-class evaluation & hardware helpers
├── notebooks/
│   ├── 01_foundational_trees_and_bagging.ipynb                     # CART, Random Forest, ExtraTrees & GPU XGBRF
│   ├── 02_classical_boosting_and_histograms.ipynb                  # AdaBoost, Friedman GBM & HistGradientBoosting
│   ├── 03_production_big_three_deep_dive.ipynb                     # XGBoost 2.x, LightGBM, CatBoost & DART on GPU
│   ├── 04_interpretable_glassbox_trees.ipynb                       # EBM (Microsoft InterpretML), RuleFit & FIGS
│   ├── 05_specialized_production_and_streaming_trees.ipynb         # Google YDF, NGBoost, River Hoeffding Trees, Isolation Forest
│   ├── 06_neural_tree_hybrids_and_differentiable_trees.ipynb       # TabNet, NODE, NCART & Differentiable Trees
│   └── 07_tabular_foundation_models_and_production_stacking.ipynb  # TabPFN v2/v3, In-Context Learning, Stacking & Treelite
├── data/                                                           # Compressed Parquet dataset storage
└── legacy_benchmark/                                               # Historical November 2025 synthetic 50k benchmark files
```

### Notebook Overview

| Notebook | Focus & Models | Key Architectural Concepts |
|---|---|---|
| [01_foundational_trees_and_bagging.ipynb](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/trees/notebooks/01_foundational_trees_and_bagging.ipynb) | CART, Random Forest, ExtraTrees, `XGBRFClassifier` | Gini impurity, cross-entropy, cost-complexity pruning (`ccp_alpha`), variance reduction math $\rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$, drop-in GPU Random Forest. |
| [02_classical_boosting_and_histograms.ipynb](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/trees/notebooks/02_classical_boosting_and_histograms.ipynb) | AdaBoost (SAMME), Friedman GBM, `HistGradientBoosting` | Multi-class residual fitting, 256-bin integer histogram discretization, histogram subtraction trick $\mathcal{H}_R = \mathcal{H}_P - \mathcal{H}_L$, native `NaN` routing. |
| [03_production_big_three_deep_dive.ipynb](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/trees/notebooks/03_production_big_three_deep_dive.ipynb) | XGBoost 2.x+, LightGBM, CatBoost, DART | 2nd-order Taylor objective $g_i f_t + \frac{1}{2} h_i f_t^2$, GOSS/EFB, Oblivious (symmetric) trees with SIMD bitwise lookups, ordered target statistics, DART tree dropout. |
| [04_interpretable_glassbox_trees.ipynb](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/trees/notebooks/04_interpretable_glassbox_trees.ipynb) | EBM (InterpretML), RuleFit (`imodels`), FIGS | Generalized Additive Models ($\text{GA}^2\text{M}$) with pairwise interaction splines, exact feature shape functions without SHAP approximations, SQL-exportable decision rules. |
| [05_specialized_production_and_streaming_trees.ipynb](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/trees/notebooks/05_specialized_production_and_streaming_trees.ipynb) | Google YDF, NGBoost, Hoeffding Trees (`river`), Isolation Forest | Sub-microsecond C++ serving, Natural Gradient in Riemannian parameter space $\mathcal{I}(\theta)^{-1} \nabla L$, real-time Kafka online learning via Hoeffding bound $\epsilon = \sqrt{\frac{R^2 \ln(1/\delta)}{2n}}$, tree path anomaly isolation. |
| [06_neural_tree_hybrids_and_differentiable_trees.ipynb](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/trees/notebooks/06_neural_tree_hybrids_and_differentiable_trees.ipynb) | TabNet, NODE, NCART, GRANDE | Sequential sparse attention mask $M[i] = \text{Sparsemax}(P[i-1] \cdot h(a))$, differentiable oblivious trees via Entmax relaxations, end-to-end backpropagation on tabular data. |
| [07_tabular_foundation_models_and_production_stacking.ipynb](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/trees/notebooks/07_tabular_foundation_models_and_production_stacking.ipynb) | TabPFN v2/v3, Stacking Ensembles, Treelite | In-Context Tabular Learning (ICL) over synthetic SEM priors, zero-shot single forward pass inference, multi-level GBDT+TFM stacking, compiling trees to standalone C code. |

---

## 4. Hardware Optimization (NVIDIA RTX 2050 4GB VRAM & CPU Vectorization)

All notebooks are pre-configured to adapt automatically to available hardware:

* **GPU Acceleration (CUDA)**:
  * **XGBoost 2.x+**: `tree_method='hist'`, `device='cuda'`, `objective='multi:softprob'`
  * **GPU Random Forest**: `xgb.XGBRFClassifier(device='cuda')`
  * **CatBoost**: `task_type='GPU'`, `loss_function='MultiClass'`
  * **PyTorch Hybrids (TabNet, NODE)**: Auto-routed to `cuda` with micro-batch sizes ($512$) budgeted to avoid CUDA Out-Of-Memory (OOM) on 4GB VRAM.
  * **TabPFN**: Accelerated in-context attention via CUDA.
* **CPU Optimization**:
  * Seamlessly patches Scikit-Learn via **Intel Extension for Scikit-learn** (`sklearnex.patch_sklearn()`) for AVX-512 hardware-accelerated CPU execution.

---

## 5. Dataset Generation & Storage Architecture

The synthetic generator [data_generator.py](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/trees/data_generator.py) streams realistic bilingual transactions into **Apache Parquet with Snappy/ZSTD compression**:

```bash
# Generate small 50k benchmark sample for instant notebook loading
python data_generator.py --test-sample

# Generate full enterprise 3,000,000 rows x 2,000 features (~500 MB on disk)
python data_generator.py --rows 3000000 --features 2000 --chunk-size 100000 --output data/ecommerce_20class_3m.parquet
```

* **Memory Efficiency**: Streaming chunk-by-chunk prevents RAM exhaustion during dataset synthesis.
* **Dictionary Encoding**: String categoricals (Arabic governorates, English channels) are stored as lightweight dictionary keys.

---

## 6. Comprehensive 2026 Tabular Decision Matrix

| Scenario / Production Constraint | Recommended Architecture | Reason & Key Advantage |
|---|---|---|
| **Ultra-Low Latency (< 10 μs microservices)** | **Google YDF** or **Treelite C-Compiled GBDT** | Zero Python runtime overhead, pure compiled C++ / SIMD execution. |
| **Large-Scale Tabular Datasets (> 100k rows)** | **LightGBM** or **XGBoost 2.x (GPU)** | GOSS and GPU histogram binning deliver unmatched training throughput. |
| **High-Cardinality Arabic/English Categories** | **CatBoost (GPU)** | Oblivious trees + ordered target statistics prevent target leakage. |
| **Strict Regulatory Audits (Credit, Health, Tax)**| **Explainable Boosting Machine (EBM)** | Exact $\text{GA}^2\text{M}$ additive splines provide 100% transparent glass-box equations. |
| **Real-Time Streaming Pipelines (Kafka/Flink)** | **Hoeffding Trees / EFDT (`river`)** | Online sample-by-sample learning with zero model retraining. |
| **Probabilistic Risk & Uncertainty Quantification**| **NGBoost** | Natural gradient descent outputs full predictive probability distributions. |
| **Cold-Start & Small Datasets (< 10k rows)** | **TabPFN (Tabular Foundation Model)** | In-context single-pass Bayesian inference with zero tuning. |
| **Leaderboard SOTA (Kaggle / Maximum Accuracy)** | **Multi-Level Stacking Ensemble** | Blending orthogonal inductive biases of GBDTs + TFMs eliminates model blindspots. |

---

## 7. Historical Legacy Baseline (November 2025 Run)

Before expanding to this comprehensive 20-class curriculum, the repository ran a historical synthetic binary classification benchmark (50,000 samples, 23 features) preserved in [trees/legacy_benchmark/](file:///c:/Users/MSI/OneDrive/Desktop/work/AI-Data-Science-Lab/trees/legacy_benchmark):

| Model | Training Time (s) | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|---|
| **XGBoost** | 2.32s | **0.9636** | **0.9640** | **0.9915** |
| **LightGBM** | **0.26s** | 0.9599 | 0.9604 | 0.9902 |
| **CatBoost** | 4.59s | 0.9626 | 0.9630 | 0.9905 |
| **HistGradientBoosting** | 0.81s | 0.9596 | 0.9599 | 0.9902 |

*The new 2026 curriculum expands far beyond this initial test, bringing 20-class multi-class complexity, GPU acceleration, explainability, streaming trees, and tabular foundation models into a unified laboratory.*