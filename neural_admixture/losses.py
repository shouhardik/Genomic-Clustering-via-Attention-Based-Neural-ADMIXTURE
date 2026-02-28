"""
Loss functions and evaluation metrics for Neural ADMIXTURE.

Loss (Eq. 2 in the paper):
    L_N(Q,F) = BCE(x, x̃) + λ ‖θ_encoder‖²_F

Evaluation metrics:
    RMSE(Q, Q_GT)  — Eq. 5
    RMSE(F, F_GT)  — Eq. 6
    Δ(Q, Q_GT)     — Eq. 7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import permutations
from typing import List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def bce_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    Element-wise binary cross-entropy between input x ∈ {0, 0.5, 1}
    and reconstruction x_hat ∈ (0, 1).

    Equivalent to ADMIXTURE's negative log-likelihood (up to factor 1/2).
    """
    return F.binary_cross_entropy_with_logits(x_hat, x, reduction="mean")


def encoder_l2_penalty(model: nn.Module) -> torch.Tensor:
    """Frobenius norm of all encoder parameters (for L2 regularization)."""
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if "encoder" in name or "bottleneck" in name:
            penalty = penalty + torch.sum(param ** 2)
    return penalty


def single_head_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    model: nn.Module,
    lam: float = 0.0,
) -> torch.Tensor:
    """
    L_N(Q, F) = BCE(x, x̃) + λ ‖θ‖²_F

    Parameters
    ----------
    x      : (N, M)  input genotypes in {0, 0.5, 1}
    x_hat  : (N, M)  reconstruction (raw logits)
    model  : NeuralADMIXTURE or similar
    lam    : regularization strength λ
    """
    loss = bce_loss(x, x_hat)
    if lam > 0:
        loss = loss + lam * encoder_l2_penalty(model)
    return loss


def multi_head_loss(
    x: torch.Tensor,
    x_hats: List[torch.Tensor],
    model: nn.Module,
    lam: float = 0.0,
) -> torch.Tensor:
    """
    L_MNA = Σ_h L_N(Q_Kh, F_Kh)   (Eq. 4)
    """
    total = torch.tensor(0.0, device=x.device)
    for x_hat in x_hats:
        total = total + bce_loss(x, x_hat)
    if lam > 0:
        total = total + lam * encoder_l2_penalty(model)
    return total


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def rmse_Q(Q: torch.Tensor, Q_gt: torch.Tensor) -> float:
    """
    RMSE(Q, Q_GT) = (1 / sqrt(N*K)) * ‖Q - Q_GT‖_F    (Eq. 5)

    Q and Q_gt must already be column-aligned (permuted to match).
    """
    N, K = Q.shape
    return (torch.norm(Q - Q_gt, p="fro") / (N * K) ** 0.5).item()


@torch.no_grad()
def rmse_F(F_est: torch.Tensor, F_gt: torch.Tensor) -> float:
    """
    RMSE(F, F_GT) = (1 / sqrt(K*M)) * ‖F - F_GT‖_F    (Eq. 6)

    F_est, F_gt: (K, M).  Must be row-aligned.
    """
    K, M = F_est.shape
    return (torch.norm(F_est - F_gt, p="fro") / (K * M) ** 0.5).item()


@torch.no_grad()
def delta_metric(Q: torch.Tensor, Q_gt: torch.Tensor) -> float:
    """
    Δ(Q, Q_GT) = (1/N²) ‖Q Q^T - Q_GT Q_GT^T‖²_F    (Eq. 7)

    Permutation-invariant measure of agreement between Q estimates.
    """
    N = Q.shape[0]
    cov_est = Q @ Q.T
    cov_gt = Q_gt @ Q_gt.T
    return (torch.norm(cov_est - cov_gt, p="fro") ** 2 / N ** 2).item()


# ---------------------------------------------------------------------------
# Permutation alignment
# ---------------------------------------------------------------------------

def permutation_align(
    Q_est: np.ndarray,
    Q_gt: np.ndarray,
    F_est: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Find the column permutation of Q_est that best matches Q_gt.

    For K <= 8, exhaustive search over all K! permutations is used.
    For K > 8, the Hungarian algorithm (scipy) solves the linear assignment
    on a cost matrix built from per-column RMSE.

    Parameters
    ----------
    Q_est : (N, K) estimated ancestry matrix.
    Q_gt : (N, K) ground-truth ancestry matrix.
    F_est : optional (K, M) allele frequency matrix to permute consistently.

    Returns
    -------
    Q_aligned : (N, K) permuted Q_est.
    F_aligned : (K, M) permuted F_est (or None if F_est was None).
    perm : (K,) the permutation applied (Q_aligned[:, i] = Q_est[:, perm[i]]).
    """
    K = Q_est.shape[1]
    assert Q_gt.shape[1] == K

    if K <= 8:
        best_perm = None
        best_cost = np.inf
        for perm in permutations(range(K)):
            perm = np.array(perm)
            cost = np.sum((Q_est[:, perm] - Q_gt) ** 2)
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
    else:
        from scipy.optimize import linear_sum_assignment

        cost_matrix = np.zeros((K, K), dtype=np.float64)
        for i in range(K):
            for j in range(K):
                cost_matrix[i, j] = np.sum((Q_gt[:, i] - Q_est[:, j]) ** 2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        best_perm = col_ind

    best_perm = np.array(best_perm)
    Q_aligned = Q_est[:, best_perm]
    F_aligned = F_est[best_perm] if F_est is not None else None
    return Q_aligned, F_aligned, best_perm
