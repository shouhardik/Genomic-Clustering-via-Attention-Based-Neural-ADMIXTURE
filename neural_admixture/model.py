"""
Neural ADMIXTURE model architecture.

Implements the autoencoder described in:
  Dominguez Mantes et al., "Neural ADMIXTURE for rapid genomic clustering",
  Nature Computational Science, 2023.

Architecture: encoder → K-dim bottleneck (softmax) → linear decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


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
# Neural ADMIXTURE
# ---------------------------------------------------------------------------

class NeuralADMIXTURE(nn.Module):
    """
    Neural ADMIXTURE autoencoder.

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
