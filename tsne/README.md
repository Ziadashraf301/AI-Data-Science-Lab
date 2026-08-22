# t-Distributed Stochastic Neighbor Embedding (t-SNE) — From Scratch

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![NumPy](<https://img.shields.io/badge/NumPy-Pure%20Implementation-013243?logo=numpy>)](https://numpy.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Validation-F7931E?logo=scikitlearn)](https://scikit-learn.org)

An end-to-end mathematical derivation and pure NumPy implementation of **t-SNE (t-Distributed Stochastic Neighbor Embedding)** (*van der Maaten & Hinton, 2008*), benchmarked and visually validated against `sklearn.manifold.TSNE` on the Wisconsin Breast Cancer dataset.

---

## 📋 Theoretical Overview

While linear dimensionality reduction techniques like **Principal Component Analysis (PCA)** project data onto global orthogonal axes of maximum variance, they fail to preserve non-linear manifold topologies and local cluster neighborhoods.

**t-SNE** resolves this by converting Euclidean distances into conditional probabilities that represent similarities:

1. **High-Dimensional Space**: Points are modeled with a **Gaussian distribution**. Close points have high $p_{ij}$, distant points have near-zero $p_{ij}$.
2. **Low-Dimensional Embedding**: Points are modeled with a heavy-tailed **Student-t distribution** (1 degree of freedom, Cauchy kernel). This eliminates the **crowding problem** by pushing moderately distant clusters apart while preserving local affinities.

---

## 🧮 Mathematical Pipeline & Algorithm Steps

```text
High-D Data X (N × D)
        │
        ▼
1. Pairwise Euclidean Distance Matrix: D_ij = ||x_i - x_j||^2
        │
        ▼
2. Binary Search per point for sigma_i matching Perplexity:
   Perp(P_i) = 2^{H(P_i)}, where H(P_i) = -sum_j p_{j|i} log2(p_{j|i})
        │
        ▼
3. Symmetrized Joint Affinities:
   p_{ij} = (p_{j|i} + p_{i|j}) / (2N)
        │
        ▼
4. Early Exaggeration: Multiply p_{ij} by 4.0 for initial iterations
        │
        ▼
5. Gradient Descent with Momentum on KL Divergence:
   KL(P || Q) = sum_i sum_j p_{ij} log(p_{ij} / q_{ij})
   dKL/dy_i = 4 * sum_j (p_{ij} - q_{ij})(y_i - y_j)(1 + ||y_i - y_j||^2)^{-1}
        │
        ▼
Low-D Embedding Y (N × 2)
```

---

## 📁 Module Structure

```text
tsne/
├── README.md                  # Detailed mathematical and empirical documentation
├── tsne_from_scratch.ipynb    # Complete NumPy implementation, step-by-step math & validation
└── tsne_comparison.png        # Comparative visualization (PCA vs. Custom t-SNE vs. Sklearn t-SNE)
```

---

## 📊 Visual Validation & Benchmark

![t-SNE Comparison](tsne_comparison.png)

### Key Takeaways:

- **PCA (Linear Projection)**: Shows overlap between malignant and benign cases along the boundary because linear projections cannot untangle non-linear relationships.
- **Custom t-SNE (NumPy)**: Produces crisp, non-linear separation between benign and malignant clusters, faithfully matching the topological geometry produced by `sklearn.manifold.TSNE`.
- **Perplexity Tuning**: Explores the trade-off between local preservation (low perplexity $\approx 5-15$) vs global geometry (high perplexity $\approx 30-50$).

## 📚 References

- **Original Paper**: L. van der Maaten and G. Hinton, *"Visualizing Data using t-SNE"*, Journal of Machine Learning Research (JMLR), 2008. [PDF Link](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
