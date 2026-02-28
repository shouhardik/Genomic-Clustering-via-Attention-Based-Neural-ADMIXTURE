"""
Neural ADMIXTURE model architecture.

Implements the autoencoder described in:
  Dominguez Mantes et al., "Neural ADMIXTURE for rapid genomic clustering",
  Nature Computational Science, 2023.

Single-head: one encoder → one (K)-dim bottleneck → one decoder
Multi-head:  shared encoder → H bottlenecks (K1…KH) → H decoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


class Encoder(nn.Module):
    """Shared encoder: BatchNorm → Linear(M→64) → GELU → Linear(64→K) → Softmax."""

    def __init__(self, n_snps: int, hidden_dim: int = 64):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_snps)
        self.fc1 = nn.Linear(n_snps, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the 64-dim hidden representation (before the K-projection)."""
        x = self.bn(x)
        x = self.fc1(x)
        x = self.act(x)
        return x


class BottleneckHead(nn.Module):
    """Projects the hidden representation to K dims with softmax → Q."""

    def __init__(self, hidden_dim: int, k: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, k)

    def forward(self, h: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = self.fc(h)
        return F.softmax(logits / temperature, dim=-1)


class Decoder(nn.Module):
    """
    Single linear layer: Q (N×K) @ F (K×M) → x̃ (N×M).

    Weights F are clamped to [0, 1] after each optimization step via
    projected gradient descent -- the caller is responsible for invoking
    `clamp_weights()` after `optimizer.step()`.
    """

    def __init__(self, k: int, n_snps: int):
        super().__init__()
        self.linear = nn.Linear(k, n_snps, bias=False)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.linear(q)

    @property
    def F_matrix(self) -> torch.Tensor:
        """Returns the K×M allele-frequency matrix (detached clone)."""
        return self.linear.weight.data.T.clone()

    def clamp_weights(self):
        """Project decoder weights into [0, 1] (projected gradient descent)."""
        with torch.no_grad():
            self.linear.weight.clamp_(0.0, 1.0)


# ---------------------------------------------------------------------------
# Single-Head Neural ADMIXTURE
# ---------------------------------------------------------------------------

class NeuralADMIXTURE(nn.Module):
    """
    Single-head Neural ADMIXTURE.

    Architecture
    ------------
    Encoder:  x → BN → Linear(M,64) → GELU → Linear(64,K) → Softmax → Q
    Decoder:  Q → Linear(K,M, no bias, weights∈[0,1]) → x̃

    Parameters
    ----------
    n_snps : int
        Number of input SNP features (M).
    k : int
        Number of ancestry clusters.
    hidden_dim : int
        Encoder hidden dimension (default 64).
    """

    def __init__(self, n_snps: int, k: int, hidden_dim: int = 64):
        super().__init__()
        self.n_snps = n_snps
        self.k = k
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(n_snps, hidden_dim)
        self.bottleneck = BottleneckHead(hidden_dim, k)
        self.decoder = Decoder(k, n_snps)

    def forward(
        self, x: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (N, M)  genotype matrix with values in {0, 0.5, 1}.
        temperature : softmax temperature (τ); >1 softens assignments.

        Returns
        -------
        x_hat : (N, M) reconstruction
        Q     : (N, K) ancestry proportions
        """
        h = self.encoder(x)
        q = self.bottleneck(h, temperature)
        x_hat = self.decoder(q)
        return x_hat, q

    def clamp_decoder_weights(self):
        self.decoder.clamp_weights()

    def get_F(self) -> torch.Tensor:
        """Return the K×M allele frequency matrix."""
        return self.decoder.F_matrix

    def encode(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Inference-only: return Q for new data."""
        h = self.encoder(x)
        return self.bottleneck(h, temperature)

    def init_decoder(self, F_init: torch.Tensor):
        """
        Initialize decoder weights from a K×M matrix (e.g. from PCK-means).
        F_init shape: (K, M).
        """
        assert F_init.shape == (self.k, self.n_snps), (
            f"Expected ({self.k}, {self.n_snps}), got {F_init.shape}"
        )
        with torch.no_grad():
            self.decoder.linear.weight.copy_(F_init.T)
            self.decoder.clamp_weights()


# ---------------------------------------------------------------------------
# Multi-Head Neural ADMIXTURE
# ---------------------------------------------------------------------------

class MultiHeadNeuralADMIXTURE(nn.Module):
    """
    Multi-head Neural ADMIXTURE.

    A shared encoder produces a 64-dim representation that is independently
    projected by H heads, each with its own K_h-dim bottleneck + decoder.
    This allows simultaneous computation of cluster assignments for
    multiple values of K in a single forward pass.

    Parameters
    ----------
    n_snps : int
        Number of input SNP features (M).
    k_values : list of int
        List of K values, one per head  (e.g. [2, 3, 4, 5, 6]).
    hidden_dim : int
        Encoder hidden dimension (default 64).
    """

    def __init__(
        self, n_snps: int, k_values: List[int], hidden_dim: int = 64
    ):
        super().__init__()
        self.n_snps = n_snps
        self.k_values = k_values
        self.n_heads = len(k_values)
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(n_snps, hidden_dim)

        self.bottlenecks = nn.ModuleList(
            [BottleneckHead(hidden_dim, k) for k in k_values]
        )
        self.decoders = nn.ModuleList(
            [Decoder(k, n_snps) for k in k_values]
        )

    def forward(
        self, x: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Returns
        -------
        x_hats : list of (N, M) reconstructions, one per head
        Qs     : list of (N, K_h) ancestry proportion matrices
        """
        h = self.encoder(x)

        x_hats = []
        qs = []
        for bottleneck, decoder in zip(self.bottlenecks, self.decoders):
            q = bottleneck(h, temperature)
            x_hat = decoder(q)
            qs.append(q)
            x_hats.append(x_hat)

        return x_hats, qs

    def clamp_decoder_weights(self):
        for decoder in self.decoders:
            decoder.clamp_weights()

    def get_F(self, head_idx: int) -> torch.Tensor:
        return self.decoders[head_idx].F_matrix

    def encode(
        self, x: torch.Tensor, temperature: float = 1.0
    ) -> List[torch.Tensor]:
        """Inference-only: return list of Q matrices for new data."""
        h = self.encoder(x)
        return [bn(h, temperature) for bn in self.bottlenecks]

    def init_decoder(self, head_idx: int, F_init: torch.Tensor):
        """Initialize a specific head's decoder from a K×M matrix."""
        k = self.k_values[head_idx]
        assert F_init.shape == (k, self.n_snps), (
            f"Head {head_idx}: expected ({k}, {self.n_snps}), got {F_init.shape}"
        )
        with torch.no_grad():
            self.decoders[head_idx].linear.weight.copy_(F_init.T)
            self.decoders[head_idx].clamp_weights()
