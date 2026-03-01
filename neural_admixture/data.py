"""
Data loading and preprocessing utilities for Neural ADMIXTURE.

Supports:
  - VCF file parsing (via cyvcf2 or scikit-allel as fallback on Windows)
  - PLINK binary .bed/.bim/.fam loading (via pandas-plink)
  - LD pruning (sliding-window pairwise r² filtering)
  - Stratified train/test splitting
  - Ground-truth Q matrix construction from population labels
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# VCF loading
# ---------------------------------------------------------------------------

def _load_vcf_cyvcf2(
    vcf_path: str,
    max_snps: Optional[int],
    maf_threshold: float,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load VCF using cyvcf2 (fast, C-based; requires htslib)."""
    from cyvcf2 import VCF

    vcf = VCF(vcf_path)
    sample_ids = list(vcf.samples)

    genotypes: List[np.ndarray] = []
    snp_ids: List[str] = []

    for variant in vcf:
        if not variant.is_snp:
            continue
        gt = np.array(variant.genotypes)[:, :2]  # (N, 2) allele calls
        dosage = gt.sum(axis=1).astype(np.float32)
        dosage[dosage < 0] = np.nan

        af = np.nanmean(dosage) / 2.0
        maf = min(af, 1.0 - af)
        if maf < maf_threshold:
            continue

        genotypes.append(dosage / 2.0)
        snp_ids.append(
            f"{variant.CHROM}:{variant.POS}:{variant.REF}:{variant.ALT[0]}"
        )

        if max_snps is not None and len(genotypes) >= max_snps:
            break

    vcf.close()
    X = np.column_stack(genotypes)
    return X, sample_ids, snp_ids


def _load_vcf_allel(
    vcf_path: str,
    max_snps: Optional[int],
    maf_threshold: float,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load VCF using scikit-allel (pure-Python parser; works on Windows)."""
    import allel

    fields = ["samples", "calldata/GT", "variants/CHROM",
              "variants/POS", "variants/REF", "variants/ALT",
              "variants/is_snp"]
    data = allel.read_vcf(vcf_path, fields=fields)

    if data is None:
        raise RuntimeError(f"scikit-allel could not read VCF: {vcf_path}")

    sample_ids = list(data["samples"])
    gt = data["calldata/GT"]          # (n_variants, n_samples, ploidy)
    is_snp = data["variants/is_snp"]
    chroms = data["variants/CHROM"]
    positions = data["variants/POS"]
    refs = data["variants/REF"]
    alts = data["variants/ALT"][:, 0]  # first ALT allele

    genotypes: List[np.ndarray] = []
    snp_ids: List[str] = []

    for i in range(gt.shape[0]):
        if not is_snp[i]:
            continue
        alleles = gt[i, :, :]                       # (n_samples, ploidy)
        dosage = alleles.sum(axis=1).astype(np.float32)
        dosage[dosage < 0] = np.nan

        af = np.nanmean(dosage) / 2.0
        maf = min(af, 1.0 - af)
        if maf < maf_threshold:
            continue

        genotypes.append(dosage / 2.0)
        snp_ids.append(f"{chroms[i]}:{positions[i]}:{refs[i]}:{alts[i]}")

        if max_snps is not None and len(genotypes) >= max_snps:
            break

    X = np.column_stack(genotypes)
    return X, sample_ids, snp_ids


def load_vcf(
    vcf_path: Union[str, Path],
    max_snps: Optional[int] = None,
    maf_threshold: float = 0.01,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load a VCF file into a genotype matrix.

    Tries **cyvcf2** first (fast, C-based). If unavailable (e.g. on Windows
    where htslib is hard to build), falls back to **scikit-allel** which
    ships a pure-Python VCF parser.

    Parameters
    ----------
    vcf_path : path to a (optionally gzipped) VCF file.
    max_snps : cap on the number of SNPs to load (None = all).
    maf_threshold : discard SNPs with minor allele frequency below this.

    Returns
    -------
    X : (N, M) float32 array with values in {0, 0.5, 1}.
    sample_ids : list of sample identifiers.
    snp_ids : list of SNP identifiers ("chrom:pos:ref:alt").
    """
    vcf_path = str(vcf_path)

    # --- try cyvcf2 first (fast, C-based) --------------------------------
    try:
        from cyvcf2 import VCF as _VCF          # noqa: F401
        X, sample_ids, snp_ids = _load_vcf_cyvcf2(
            vcf_path, max_snps, maf_threshold
        )
    except ImportError:
        # --- fallback to scikit-allel ------------------------------------
        try:
            import allel as _allel               # noqa: F401
        except ImportError:
            raise ImportError(
                "Neither cyvcf2 nor scikit-allel is installed.\n"
                "  • Linux / macOS  →  pip install cyvcf2\n"
                "  • Windows        →  pip install scikit-allel\n"
                "At least one is required for VCF loading."
            )
        print("[INFO] cyvcf2 not available – using scikit-allel fallback")
        X, sample_ids, snp_ids = _load_vcf_allel(
            vcf_path, max_snps, maf_threshold
        )

    # --- impute missing values -------------------------------------------
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(nan_mask)
        X[inds] = np.take(col_means, inds[1])

    return X.astype(np.float32), sample_ids, snp_ids


# ---------------------------------------------------------------------------
# PLINK binary loading
# ---------------------------------------------------------------------------

def load_plink(
    bed_path: Union[str, Path],
    maf_threshold: float = 0.01,
    max_snps: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load PLINK binary files (.bed, .bim, .fam).

    Parameters
    ----------
    bed_path : path to the .bed file (companion .bim/.fam must exist).
    maf_threshold : discard SNPs with minor allele frequency below this.
    max_snps : cap on the number of SNPs to load.

    Returns
    -------
    X : (N, M) float32 array with values in {0, 0.5, 1}.
    bim : variant metadata DataFrame.
    fam : sample metadata DataFrame.
    """
    try:
        from pandas_plink import read_plink
    except ImportError:
        raise ImportError(
            "pandas-plink is required for PLINK loading. "
            "Install with: pip install pandas-plink"
        )

    bim, fam, genotype = read_plink(str(bed_path).replace(".bed", ""))
    X = genotype.compute().T.astype(np.float32)  # (N, M)

    if max_snps is not None:
        X = X[:, :max_snps]
        bim = bim.iloc[:max_snps]

    X = X / 2.0

    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(nan_mask)
        X[inds] = np.take(col_means, inds[1])

    if maf_threshold > 0:
        af = X.mean(axis=0)
        maf = np.minimum(af, 1.0 - af)
        keep = maf >= maf_threshold
        X = X[:, keep]
        bim = bim.iloc[keep.nonzero()[0]]

    return X.astype(np.float32), bim, fam


# ---------------------------------------------------------------------------
# Simulated / synthetic data
# ---------------------------------------------------------------------------

def simulate_genotypes(
    n_samples_per_pop: int = 200,
    n_snps: int = 5000,
    n_populations: int = 5,
    fst: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate genotype data using the Balding-Nichols model.

    Each population k has allele frequencies drawn from
    Beta(p*(1-Fst)/Fst, (1-p)*(1-Fst)/Fst) where p ~ Uniform(0.1, 0.9).

    Parameters
    ----------
    n_samples_per_pop : samples per population.
    n_snps : number of SNPs.
    n_populations : K.
    fst : fixation index controlling population divergence.
    random_state : RNG seed.

    Returns
    -------
    X : (N, M) genotype matrix in {0, 0.5, 1}.
    Q_gt : (N, K) one-hot ground-truth ancestry matrix.
    F_gt : (K, M) ground-truth allele frequency matrix.
    labels : (N,) integer population labels.
    """
    rng = np.random.RandomState(random_state)
    N = n_samples_per_pop * n_populations
    K = n_populations
    M = n_snps

    p_ancestral = rng.uniform(0.1, 0.9, size=M)

    F_gt = np.zeros((K, M), dtype=np.float32)
    for k in range(K):
        a = p_ancestral * (1 - fst) / fst
        b = (1 - p_ancestral) * (1 - fst) / fst
        F_gt[k] = rng.beta(a, b).astype(np.float32)

    F_gt = np.clip(F_gt, 0.001, 0.999)

    X = np.zeros((N, M), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int64)

    for k in range(K):
        start = k * n_samples_per_pop
        end = (k + 1) * n_samples_per_pop
        labels[start:end] = k
        allele1 = rng.binomial(1, F_gt[k], size=(n_samples_per_pop, M))
        allele2 = rng.binomial(1, F_gt[k], size=(n_samples_per_pop, M))
        X[start:end] = (allele1 + allele2) / 2.0

    Q_gt = np.eye(K, dtype=np.float32)[labels]

    return X.astype(np.float32), Q_gt, F_gt, labels


# ---------------------------------------------------------------------------
# LD Pruning
# ---------------------------------------------------------------------------

def ld_prune(
    X: np.ndarray,
    window_size: int = 50,
    step: int = 10,
    r2_threshold: float = 0.2,
) -> np.ndarray:
    """
    LD prune a genotype matrix using a sliding-window pairwise r² filter.

    Within each window of `window_size` SNPs (shifted by `step`), remove
    one SNP from each pair whose r² exceeds `r2_threshold`.

    Parameters
    ----------
    X : (N, M) genotype matrix.
    window_size : number of SNPs per window.
    step : window slide step.
    r2_threshold : pairs above this are pruned.

    Returns
    -------
    indices : 1-D array of retained SNP column indices.
    """
    M = X.shape[1]
    remove = set()

    for start in range(0, M, step):
        end = min(start + window_size, M)
        window_idx = [j for j in range(start, end) if j not in remove]
        if len(window_idx) < 2:
            continue

        sub = X[:, window_idx]
        means = sub.mean(axis=0, keepdims=True)
        centered = sub - means
        stds = sub.std(axis=0, keepdims=True)
        stds[stds == 0] = 1.0
        normed = centered / stds

        corr = (normed.T @ normed) / X.shape[0]
        r2 = corr ** 2

        for i in range(len(window_idx)):
            if window_idx[i] in remove:
                continue
            for j in range(i + 1, len(window_idx)):
                if window_idx[j] in remove:
                    continue
                if r2[i, j] > r2_threshold:
                    remove.add(window_idx[j])

    kept = sorted(set(range(M)) - remove)
    return np.array(kept, dtype=np.int64)


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def stratified_split(
    X: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train/test split preserving population proportions.

    Returns
    -------
    X_train, X_test, labels_train, labels_test
    """
    return train_test_split(
        X, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Ground-truth Q matrix
# ---------------------------------------------------------------------------

def build_q_ground_truth(
    labels: np.ndarray,
    k: Optional[int] = None,
) -> np.ndarray:
    """
    Build a one-hot (N, K) Q_GT matrix from integer population labels.

    Parameters
    ----------
    labels : (N,) integer labels in [0, K-1].
    k : number of clusters (inferred from labels if None).

    Returns
    -------
    Q_gt : (N, K) float32 one-hot matrix.
    """
    if k is None:
        k = int(labels.max()) + 1
    return np.eye(k, dtype=np.float32)[labels]


def labels_from_populations(
    population_list: List[str],
    pop_to_superpop: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Convert a list of population strings to integer labels.

    Parameters
    ----------
    population_list : per-sample population identifiers.
    pop_to_superpop : optional mapping from sub-population to super-population
        (e.g. "GBR" -> "EUR"). If provided, labels are based on super-populations.

    Returns
    -------
    labels : (N,) integer array.
    label_map : dict mapping population name to integer.
    """
    if pop_to_superpop is not None:
        pops = [pop_to_superpop.get(p, p) for p in population_list]
    else:
        pops = list(population_list)

    unique_pops = sorted(set(pops))
    label_map = {p: i for i, p in enumerate(unique_pops)}
    labels = np.array([label_map[p] for p in pops], dtype=np.int64)
    return labels, label_map


# 1000 Genomes super-population mapping
SUPERPOP_MAP_1KG = {
    "CHB": "EAS", "JPT": "EAS", "CHS": "EAS", "CDX": "EAS", "KHV": "EAS",
    "CEU": "EUR", "TSI": "EUR", "FIN": "EUR", "GBR": "EUR", "IBS": "EUR",
    "YRI": "AFR", "LWK": "AFR", "GWD": "AFR", "MSL": "AFR", "ESN": "AFR",
    "ASW": "AFR", "ACB": "AFR",
    "MXL": "AMR", "PUR": "AMR", "CLM": "AMR", "PEL": "AMR",
    "GIH": "SAS", "PJL": "SAS", "BEB": "SAS", "STU": "SAS", "ITU": "SAS",
}
