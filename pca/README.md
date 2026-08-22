# Principal Component Analysis (PCA) — From Theory to Practice

## Breast Cancer Diagnostics Case Study

A comprehensive implementation of **PCA from scratch**, validated against scikit-learn, applied to the Wisconsin Breast Cancer dataset for multicollinearity diagnosis and dimensionality reduction.

---

## 📋 Project Overview

**Objective:**

- Diagnose severe multicollinearity in 30 clinical features
- Implement PCA from first principles (SVD & Eigenvalue decomposition)
- Validate custom implementation against sklearn
- Quantify the accuracy-vs-efficiency trade-off in classification pipelines

**Dataset:** Wisconsin Breast Cancer Dataset

- **Size:** 569 samples × 30 features
- **Target:** Binary classification (Benign / Malignant)
- **Class Balance:** 63% Benign, 37% Malignant
- **Missing Values:** None

---

## 🔄 Pipeline Architecture

```
Raw Data → EDA (Skewness/Outliers) → Correlation Clustering 
→ VIF Analysis → PowerTransformer → PCA (SVD & Eigen) 
→ Validation → Classification
```

---

## 📊 Key Sections

### 1. **Data Loading & Preprocessing**

- Minimal cleaning: removes ID column and artifact column (`Unnamed: 32`)
- Binary label encoding: M=1 (Malignant), B=0 (Benign)
- **Skewness Analysis:** 22/30 features have |skew| > 1
- **Outliers:** All 30 features contain z-score outliers (|z| > 3)
- **PowerTransformer (Yeo-Johnson):** Applied to handle non-normal distributions
  - Improves recall for malignant class from 0.8810 → 0.9524

### 2. **Exploratory Data Analysis (EDA)**

- Distribution analysis (histograms with seaborn)
- Skewness and kurtosis metrics
- Outlier detection using z-scores
- Class balance assessment

### 3. **Multicollinearity Diagnosis**

- **Pearson Correlation Matrix:** Identifies highly correlated feature pairs
- **Hierarchical Clustering:** Groups correlated features using dendrogram
- **Variance Inflation Factor (VIF):** Quantifies multicollinearity severity
  - Features with VIF > 10 indicate severe multicollinearity
  - Many features exceed this threshold before PCA

### 4. **Principal Component Analysis Implementation**

#### From Scratch (Two Methods):

1. **SVD-Based PCA:** Direct singular value decomposition
2. **Eigenvalue-Based PCA:** Covariance matrix eigendecomposition

#### Key Outputs:

- Principal components (loadings/weights)
- Explained variance ratio per component
- Cumulative explained variance
- Transformed data in reduced space

#### Validation:

- Custom implementation validated against `sklearn.decomposition.PCA`
- Numerical equivalence verified for transformed data and explained variance

### 5. **Explained Variance Analysis**

- Scree plot: cumulative explained variance vs. number of components
- **Optimal component selection:** 3 components retain 75% of total variance
- Trade-off analysis: Just 3 components reduce dimensions by 90% while retaining most information

### 6. **Classification Pipeline**

#### Models Evaluated:

- Logistic Regression with 3 PCA components versus 30 original features
- Pipeline: `StandardScaler → PowerTransformer → PCA → LogisticRegression`

#### Metrics Computed:

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
- Cross-validation scores (stratified k-fold)

---

## 📁 File Structure

```
pca/
├── README.md                        # This file
├── PCA.ipynb                        # Main Jupyter notebook with full analysis
└── breast_cancer_dataset.csv        # Wisconsin Breast Cancer Dataset (569 × 30)
```

---

## 🛠️ Dependencies

- **numpy**: Numerical computations (SVD, eigendecomposition)
- **pandas**: Data manipulation and analysis
- **scikit-learn**: PCA, preprocessing, classification, metrics
- **scipy**: Statistics, clustering, distance metrics
- **statsmodels**: Variance Inflation Factor (VIF)
- **matplotlib & seaborn**: Visualization (plots, dendrograms, heatmaps)

---

## 📈 Results Summary

### Classification Performance (Logistic Regression with PowerTransformer)

| Metric                | Without PCA (30 features) | With PCA (3 components) | Difference |
| --------------------- | ------------------------- | ----------------------- | ---------- |
| Accuracy              | 0.9825 (98.25%)           | 0.9737 (97.37%)         | −0.88%    |
| Precision (Malignant) | 0.9722                    | 0.9608                  | −1.14%    |
| Recall (Malignant)    | **0.9524**          | **0.9524**        | ✅ Tied    |
| F1-Score (Malignant)  | 0.9621                    | 0.9565                  | −0.56%    |
| ROC-AUC               | 0.9983                    | 0.9974                  | −0.09%    |

### Efficiency Gains (PCA with 3 components)

| Metric            | Without PCA | With PCA (3) | Improvement              |
| ----------------- | ----------- | ------------ | ------------------------ |
| Features Retained | 30          | 3            | **90% reduction**  |
| Variance Captured | 100%        | 75.01%       | —                       |
| Memory Usage      | 106.6 KB    | 10.7 KB      | **10× reduction** |
| Training Time     | baseline    | ~3× faster  | **~3× speedup**   |

**Key Finding:** After PowerTransformer (Yeo-Johnson) correction, PCA achieves **10× compression with <1% accuracy cost and tied recall** for malignant classification.

---

## 🎯 Learning Objectives

By working through this notebook, you will learn:

1. ✅ Why multicollinearity is problematic (VIF analysis)
2. ✅ How PCA works mathematically (SVD vs. Eigenvalue methods)
3. ✅ How to implement PCA from scratch
4. ✅ How to apply data transformations for non-normal distributions
5. ✅ How to validate custom implementations against established libraries
6. ✅ How to evaluate dimensionality reduction impact on classifier performance

---

## 💡 Key Insights

- **Multicollinearity:** Original 30 features exhibit severe multicollinearity (VIF up to 63,306), but PCA effectively decorrelates them
- **Normality Matters:** PowerTransformer (Yeo-Johnson) correction is critical—it improved recall from 0.8810 → 0.9524 when combined with PCA, closing the gap entirely with the no-PCA baseline
- **Optimal Compression:** Just **3 components capture 75% of variance** while reducing memory by 10× and training time by ~3×
- **Accuracy-Recall Trade-off:** <1% accuracy loss (98.25% → 97.37%) while maintaining equal recall on malignant cases—excellent trade-off for production efficiency

---

## 📚 References

- [PCA Theory: Pattern Recognition and Machine Learning](https://www.springer.com/gp/book/9780387310732)
- [Scikit-learn PCA Documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [UCI Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
