"""
Decoder initialization strategies for Neural ADMIXTURE.

PCK-means (PCA + K-means):
  1. Run PCA on the (N, M) genotype matrix to get a low-dim embedding.
  2. Run K-means on the PCA scores to find K cluster centers.
  3. Map centers back to the original M-dim space → use as initial F.
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Optional


def pck_means_init(
    X: np.ndarray,
    k: int,
    n_pca_components: int = 20,
    random_state: int = 42,
) -> torch.Tensor:
    """
    PCK-means initialization for the decoder weight matrix F.

    Steps
    -----
    1. PCA to `n_pca_components` dimensions.
    2. K-means (K clusters) in PCA space.
    3. Compute cluster centroids in the original SNP space.
    4. Clamp centroids to [0, 1].

    Parameters
    ----------
    X : (N, M) numpy array of genotypes in {0, 0.5, 1}.
    k : number of clusters.
    n_pca_components : PCA dimensions for K-means.
    random_state : reproducibility seed.

    Returns
    -------
    F_init : (K, M) torch.Tensor with values in [0, 1].
    """
    n_components = min(n_pca_components, X.shape[0], X.shape[1])

    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(Z)

    centroids = np.zeros((k, X.shape[1]), dtype=np.float32)
    for c in range(k):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = X[mask].mean(axis=0)
        else:
            centroids[c] = X[np.random.randint(X.shape[0])].copy()

    centroids = np.clip(centroids, 0.0, 1.0)
    return torch.from_numpy(centroids)


def random_init(k: int, n_snps: int) -> torch.Tensor:
    """Uniform random initialization in [0, 1] as a simple baseline."""
    return torch.rand(k, n_snps)
