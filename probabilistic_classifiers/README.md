# Probabilistic & Discriminative Classification From Scratch

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![NumPy](<https://img.shields.io/badge/NumPy-Pure%20Math-013243?logo=numpy>)](https://numpy.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Evaluation-F7931E?logo=scikitlearn)](https://scikit-learn.org)

A complete mathematical and empirical study of **four foundational classification paradigms**, implemented strictly from first principles using **pure NumPy** (no `sklearn` estimators):

- **Linear Discriminant Analysis (LDA)**
- **Quadratic Discriminant Analysis (QDA)**
- **Gaussian Naive Bayes (GNB)**
- **Logistic Regression (LogReg)**

---

## 📐 Mathematical Framework & Assumptions Comparison

| Model                  | Generative / Discriminative | Covariance Matrix$\mathbf{\Sigma}_k$ Structure                                               | Decision Boundary Geometry | Parameters to Estimate            |
| ---------------------- | --------------------------- | ---------------------------------------------------------------------------------------------- | -------------------------- | --------------------------------- |
| **LDA**          | Generative                  | **Shared**: $\mathbf{\Sigma}_0 = \mathbf{\Sigma}_1 = \mathbf{\Sigma}$                  | Linear Hyperplane          | $p + \frac{p(p+1)}{2}$          |
| **QDA**          | Generative                  | **Class-Specific**: $\mathbf{\Sigma}_0 \neq \mathbf{\Sigma}_1$                         | Quadratic Surface (Conic)  | $2p + 2 \cdot \frac{p(p+1)}{2}$ |
| **Naive Bayes**  | Generative                  | **Diagonal**: $\mathbf{\Sigma}_k = \text{diag}(\sigma_{k,1}^2, \dots, \sigma_{k,p}^2)$ | Axis-Aligned Quadratic     | $2p + 2p = 4p$                  |
| **Logistic Reg** | Discriminative              | Non-parametric (No Gaussian assumption)                                                        | Linear Hyperplane          | $p + 1$                         |

---

## 🔬 Experimental Regimes & Key Insights

### 1. The Single Feature Regime ($p = 1$)

Using `radius_mean` as an intuitive 1D example to observe how different mathematical assumptions dictate the exact threshold cutoff:

- **LDA** creates a single threshold balancing Mahalanobis distances under pooled variance.
- **QDA** creates two split points or curved quadratic boundaries if class variances differ significantly ($\sigma_{\text{mal}} = 3.20$ vs $\sigma_{\text{ben}} = 1.78$).
- **Logistic Regression** fits the smooth sigmoid transition.

---

### 2. High-Dimensional Regime ($p = 30$) & Cross-Validation

Evaluated on the full Wisconsin Breast Cancer dataset (569 samples $\times$ 30 features):

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ Model                          │ Accuracy │ Recall (Mal) │ ROC-AUC │ Param Count│
├────────────────────────────────┼──────────┼──────────────┼─────────┼────────────┤
│ Linear Discriminant (LDA)      │ 95.96%   │ 90.09%       │ 0.9892  │ 495 params │
│ Quadratic Discriminant (QDA)   │ 96.31%   │ 92.45%       │ 0.9914  │ 960 params │
│ Gaussian Naive Bayes (GNB)     │ 93.85%   │ 89.62%       │ 0.9856  │ 120 params │
│ Logistic Regression (scratch)  │ 97.89%   │ 95.75%       │ 0.9958  │ 31 params  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. The Small Sample Breakdown Regime ($n < 30$)

What happens when sample size $n$ approaches or falls below feature dimensionality $p$?

- **QDA Breakdown**: When $n_k \le p$, the empirical class covariance matrix $\mathbf{\Sigma}_k$ becomes rank-deficient (singular), causing determinant $\det(\mathbf{\Sigma}_k) \to 0$ and inversion $\mathbf{\Sigma}_k^{-1}$ to explode without regularization.
- **GNB Robustness**: Because GNB only estimates $2p$ diagonal parameters, it is immune to matrix inversion singularities and dominates when sample sizes are tiny ($n \in [10, 30]$).
- **LDA Stability**: Benefits from sample pooling ($n_0 + n_1 - 2$ degrees of freedom), remaining stable longer than QDA.

---

## 📁 Module Structure

```text
probabilistic_classifiers/
├── README.md                          # Theoretical guide and experimental findings
└── probabilistic_classifiers.ipynb   # Pure NumPy implementation, proofs, plots & CV
```
