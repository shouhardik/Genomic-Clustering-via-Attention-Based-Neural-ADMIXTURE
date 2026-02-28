# Neural ADMIXTURE — CSE 284 Project

Genomic Clustering via Attention-Based Neural ADMIXTURE.

A from-scratch implementation of the Neural ADMIXTURE framework for rapid
estimation of ancestry proportions from genotype data. The model uses an
autoencoder whose decoder weights directly encode the allele-frequency matrix
**F**, while the bottleneck produces per-individual ancestry fractions **Q**.

> Dominguez Mantes et al., "Neural ADMIXTURE for rapid genomic clustering",
> *Nature Computational Science*, 2023.

---

## Architecture

### Single-Head (one value of K)

```
x (N×M)
  → BatchNorm
  → Linear(M → 64) → GELU          ← shared encoder
  → Linear(64 → K) → Softmax        → Q (N×K)   [ancestry proportions]
  → Linear(K → M, weights ∈ [0,1])  → x̃ (N×M)  [reconstruction]
```

The decoder weight matrix **F** (K × M) is the allele-frequency matrix — each
row is a cluster centroid in SNP-frequency space. Weights are clamped to [0, 1]
via projected gradient descent after every optimizer step.

### Multi-Head (multiple K values simultaneously)

A shared encoder (up to the 64-dim hidden layer) feeds into **H** independent
heads, each with its own K_h-dim bottleneck and decoder. This lets you train
K = 2, 3, 4, 5, 6 simultaneously in a single run, amortizing the encoder cost.

### Loss Function

```
L(Q, F) = BCE(x, x̃)  +  λ ‖θ_encoder‖²_F
```

Binary cross-entropy on the reconstruction (equivalent to the ADMIXTURE negative
log-likelihood up to a factor of ½) plus L2 regularization on the encoder
weights to soften cluster assignments.

---

## Project Structure

```
├── neural_admixture/             # Core library
│   ├── __init__.py               # Public API exports
│   ├── model.py                  # NeuralADMIXTURE & MultiHeadNeuralADMIXTURE
│   ├── losses.py                 # BCE loss, L2 reg, RMSE / Δ metrics
│   ├── initialization.py        # PCK-means decoder initialization
│   ├── data.py                   # VCF / PLINK loading, LD pruning, simulation
│   ├── trainer.py                # Training loop, inference, evaluation
│   ├── visualization.py          # PCA plots, admixture bar plots, training curves
│   └── benchmark.py              # Wall-clock timing & peak-memory profiling
│
├── experiments/
│   ├── run_experiment.ipynb      # End-to-end experiment notebook
│   ├── single_head_k5.pt        # Saved single-head checkpoint (K = 5)
│   └── multi_head_k2to6.pt      # Saved multi-head checkpoint (K = 2–6)
│
├── data/
│   └── 1kg/                      # 1000 Genomes Phase 3 (chr22)
│       ├── ALL.chr22.phase3.vcf.gz
│       ├── ALL.chr22.phase3.vcf.gz.tbi
│       └── 1kg_panel.tsv         # Sample metadata (2 504 samples)
│
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone <repo-url>
cd Genomic-Clustering-via-Attention-Based-Neural-ADMIXTURE
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `torch >= 2.0` | Model, autograd, GPU support |
| `numpy >= 1.24` | Array operations |
| `scikit-learn >= 1.3` | PCA, K-means, train/test splitting |
| `matplotlib >= 3.7` | Visualization |
| `scipy >= 1.11` | Hungarian algorithm for permutation alignment |
| `cyvcf2 >= 0.30` | Fast VCF parsing |
| `pandas-plink >= 2.2` | PLINK binary file loading |
| `tqdm >= 4.65` | Progress bars |
| `jupyter >= 1.0` | Notebook environment |

---

## Quick Start

### Single-Head

```python
from neural_admixture import NeuralADMIXTURE, Trainer

# X_train: (N, M) array with values in {0, 0.5, 1}
model = NeuralADMIXTURE(n_snps=5000, k=5)
trainer = Trainer(model, lr=1e-3, lam=0.0005, batch_size=256)

trainer.initialize_decoders(X_train)      # PCK-means init
trainer.fit(X_train, n_epochs=50)         # train

Q = trainer.predict(X_train)              # (N, 5) ancestry proportions
F = model.get_F()                         # (5, M) allele-frequency matrix
```

### Multi-Head

```python
from neural_admixture import MultiHeadNeuralADMIXTURE, Trainer

model = MultiHeadNeuralADMIXTURE(n_snps=5000, k_values=[2, 3, 4, 5, 6])
trainer = Trainer(model, lr=1e-3, batch_size=256)

trainer.initialize_decoders(X_train)
trainer.fit(X_train, n_epochs=50)

Qs = trainer.predict(X_train)  # list of 5 arrays: (N,2), (N,3), …, (N,6)
```

---

## Data Pipeline

The `data` module supports multiple input formats and simulation:

```python
from neural_admixture import load_vcf, load_plink, simulate_genotypes, ld_prune, stratified_split

# Load real data
X, samples = load_vcf("data/1kg/ALL.chr22.phase3.vcf.gz", maf_threshold=0.05)

# Or simulate (Balding-Nichols model)
X, Q_true, F_true = simulate_genotypes(
    n_samples=1000, n_snps=10000, n_pops=5, fst=0.1
)

# Preprocess
X_pruned = ld_prune(X, window=50, step=10, r2_thresh=0.2)
X_train, X_test, idx_train, idx_test = stratified_split(X_pruned, labels, test_size=0.2)
```

### Supported Formats

| Format | Loader | Notes |
|---|---|---|
| VCF (.vcf.gz) | `load_vcf` | Uses cyvcf2; supports MAF filtering |
| PLINK (.bed/.bim/.fam) | `load_plink` | Uses pandas-plink |
| Simulated | `simulate_genotypes` | Balding-Nichols model with configurable Fst |

---

## Running Experiments

Open `experiments/run_experiment.ipynb` in Jupyter and run cells sequentially. The notebook covers:

1. **Simulated data** — Balding-Nichols model (5 populations, 10K SNPs, Fst = 0.1)
2. **Single-head training** — K = 5 with PCK-means init, 50 epochs
3. **Multi-head training** — K = 2 through 6 with a shared encoder
4. **Evaluation** — RMSE(Q), RMSE(F), Δ metric with permutation alignment
5. **Visualization** — PCA projections, stacked bar plots, training curves
6. **Benchmarking** — CPU vs GPU timing and memory profiling
7. **Real data (1000 Genomes)** — chr22, 2 504 samples, 5 super-populations
8. **Real data (SGDP)** — 279 high-coverage genomes, 130 populations (optional)

Pre-trained checkpoints are provided in `experiments/` so you can skip
training and go straight to evaluation/visualization.

---

## Evaluation Metrics

| Metric | Formula | Description |
|---|---|---|
| **RMSE(Q)** | √(mean((Q̂ − Q_gt)²)) | Reconstruction error of ancestry proportions |
| **RMSE(F)** | √(mean((F̂ − F_gt)²)) | Reconstruction error of allele frequencies |
| **Δ** | max_k \|q̂_k − q_gt_k\| averaged over samples | Permutation-invariant worst-case ancestry error |

Columns of Q̂ are automatically aligned to the ground truth via exhaustive
search (K ≤ 8) or the Hungarian algorithm (K > 8) before computing metrics.

---

## Visualization

```python
from neural_admixture import (
    plot_pca_with_centroids,
    plot_admixture_barplot,
    plot_multihead_barplots,
    plot_training_history,
)

plot_pca_with_centroids(X_test, Q, F)          # PCA scatter + learnt centroids
plot_admixture_barplot(Q, labels)               # STRUCTURE-style bar plot
plot_multihead_barplots(Qs, labels, [2,3,4,5,6])  # side-by-side for each K
plot_training_history(trainer.history)           # loss curves
```

---

## Benchmarking

```python
from neural_admixture import benchmark_training, benchmark_inference

train_results = benchmark_training(model, X_train, n_epochs=10)
infer_results = benchmark_inference(trainer, X_test, n_runs=50)
```

The `benchmark` module provides `timer` and `track_memory` context managers for
custom profiling, and `format_results_table` for pretty-printed comparisons
across devices (CPU / CUDA / MPS).

---

## Key Concepts

| Component | Description |
|---|---|
| **Q matrix** (N × K) | Fractional ancestry assignments per individual |
| **F matrix** (K × M) | Allele-frequency centroids per cluster |
| **BCE loss** | Binary cross-entropy ≡ ADMIXTURE neg log-likelihood (× ½) |
| **L2 reg** (λ) | Penalizes encoder weights to soften cluster assignments |
| **Softmax temp** (τ) | τ > 1 at inference softens Q; τ < 1 hardens it |
| **PCK-means** | PCA + K-means to initialize F for faster convergence |
| **Proj. grad. descent** | Clamp decoder weights to [0, 1] after each step |
| **Multi-head** | Shared encoder, independent bottleneck + decoder per K |
| **LD pruning** | Sliding-window linkage-disequilibrium filtering |
| **Permutation alignment** | Match estimated clusters to ground truth before scoring |

---

## References

- Dominguez Mantes, A., Bustamante, D., Poyatos, C. et al. "Neural ADMIXTURE
  for rapid genomic clustering." *Nat Comput Sci* **3**, 802–814 (2023).
- Alexander, D. H., Novembre, J. & Lange, K. "Fast model-based estimation of
  ancestry in unrelated individuals." *Genome Res.* **19**, 1655–1664 (2009).
- The 1000 Genomes Project Consortium. "A global reference for human genetic
  variation." *Nature* **526**, 68–74 (2015).
