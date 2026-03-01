"""
Training loop for Neural ADMIXTURE.

Handles:
  - Mini-batch training with Adam optimizer
  - Projected gradient descent (decoder weight clamping after each step)
  - Optional L2 regularization on encoder weights
  - Logging of loss / metrics per epoch
  - Softmax tempering at inference time
"""

import time
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Union
from tqdm import tqdm

from .model import NeuralADMIXTURE
from .losses import (
    single_head_loss,
    rmse_Q,
    rmse_F,
    delta_metric,
    permutation_align,
)
from .initialization import pck_means_init


class Trainer:
    """
    Trainer for Neural ADMIXTURE.

    Parameters
    ----------
    model : NeuralADMIXTURE
    lr : learning rate (default 1e-3)
    lam : L2 regularization strength λ (default 0)
    batch_size : mini-batch size (default 256)
    device : 'cpu', 'cuda', or 'mps'
    """

    def __init__(
        self,
        model: NeuralADMIXTURE,
        lr: float = 1e-3,
        lam: float = 0.0,
        batch_size: int = 256,
        device: Optional[str] = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.lam = lam
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.history: List[Dict[str, float]] = []

    def _make_loader(
        self, X: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        tensor = torch.from_numpy(X.astype(np.float32))
        dataset = TensorDataset(tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    # ------------------------------------------------------------------
    # PCK-means initialization helper
    # ------------------------------------------------------------------

    def initialize_decoders(
        self,
        X_train: np.ndarray,
        n_pca_components: int = 20,
        random_state: int = 42,
    ):
        """Run PCK-means and load the resulting F into the decoder."""
        F_init = pck_means_init(
            X_train, self.model.k, n_pca_components, random_state
        )
        self.model.init_decoder(F_init.to(self.device))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        n_epochs: int = 50,
        X_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Train the model.

        Parameters
        ----------
        X_train : (N, M) genotype array with values in {0, 0.5, 1}.
        n_epochs : number of training epochs.
        X_val : optional validation set.
        verbose : show progress bar.

        Returns
        -------
        history : list of per-epoch metric dicts.
        """
        loader = self._make_loader(X_train, shuffle=True)
        val_loader = self._make_loader(X_val, shuffle=False) if X_val is not None else None

        epoch_iter = tqdm(range(1, n_epochs + 1), desc="Training") if verbose else range(1, n_epochs + 1)

        for epoch in epoch_iter:
            t0 = time.time()
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                self.optimizer.zero_grad()

                x_hat, q = self.model(batch_x)
                loss = single_head_loss(batch_x, x_hat, self.model, self.lam)

                loss.backward()
                self.optimizer.step()

                self.model.clamp_decoder_weights()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            elapsed = time.time() - t0

            record = {"epoch": epoch, "train_loss": avg_loss, "time_s": elapsed}

            if val_loader is not None:
                val_loss = self._eval_loss(val_loader)
                record["val_loss"] = val_loss

            self.history.append(record)

            if verbose:
                msg = f"epoch {epoch:3d} | loss {avg_loss:.6f}"
                if "val_loss" in record:
                    msg += f" | val_loss {record['val_loss']:.6f}"
                msg += f" | {elapsed:.1f}s"
                epoch_iter.set_postfix_str(msg)

        return self.history

    @torch.no_grad()
    def _eval_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(self.device)
            x_hat, _ = self.model(batch_x)
            loss = single_head_loss(batch_x, x_hat, self.model, lam=0.0)
            total_loss += loss.item()
            n += 1
        return total_loss / n

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        X: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Produce ancestry assignments Q (N, K) for (possibly unseen) data."""
        self.model.eval()
        tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        q = self.model.encode(tensor, temperature)
        return q.cpu().numpy()

    # ------------------------------------------------------------------
    # Evaluation against ground truth
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        X: np.ndarray,
        Q_gt: Optional[np.ndarray] = None,
        F_gt: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        align: bool = True,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics with optional permutation alignment.

        Parameters
        ----------
        X : genotype array
        Q_gt : ground-truth ancestry proportions (N, K)
        F_gt : ground-truth allele frequencies (K, M)
        temperature : softmax temperature for inference
        align : if True, apply permutation alignment before computing metrics
        """
        self.model.eval()
        tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)

        Q_est_t = self.model.encode(tensor, temperature)
        F_est_t = self.model.get_F().to(self.device)

        Q_est_np = Q_est_t.cpu().numpy()
        F_est_np = F_est_t.cpu().numpy()

        if align and Q_gt is not None:
            Q_est_np, F_est_np, _ = permutation_align(
                Q_est_np, Q_gt, F_est_np
            )

        metrics: Dict[str, float] = {}

        if Q_gt is not None:
            Q_est_t = torch.from_numpy(Q_est_np).to(self.device)
            Q_gt_t = torch.from_numpy(Q_gt.astype(np.float32)).to(self.device)
            metrics["rmse_Q"] = rmse_Q(Q_est_t, Q_gt_t)
            metrics["delta"] = delta_metric(Q_est_t, Q_gt_t)

        if F_gt is not None:
            F_est_t = torch.from_numpy(F_est_np).to(self.device)
            F_gt_t = torch.from_numpy(F_gt.astype(np.float32)).to(self.device)
            metrics["rmse_F"] = rmse_F(F_est_t, F_gt_t)

        return metrics

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the trained model, optimizer state, and training history.

        The checkpoint includes enough information to reconstruct the model
        architecture and resume training or run inference on new data.
        """
        model_config = {
            "n_snps": self.model.n_snps,
            "k": self.model.k,
            "hidden_dim": self.model.hidden_dim,
        }

        checkpoint = {
            "model_config": model_config,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "lam": self.lam,
            "batch_size": self.batch_size,
        }
        torch.save(checkpoint, str(path))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None,
    ) -> "Trainer":
        """
        Load a saved checkpoint and reconstruct the Trainer.

        Parameters
        ----------
        path : path to the saved checkpoint (.pt file).
        device : device to load the model onto.

        Returns
        -------
        trainer : fully reconstructed Trainer ready for inference or
            continued training.
        """
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        config = checkpoint["model_config"]

        model = NeuralADMIXTURE(
            n_snps=config["n_snps"],
            k=config["k"],
            hidden_dim=config["hidden_dim"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        lr = checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]
        trainer = cls(
            model=model,
            lr=lr,
            lam=checkpoint["lam"],
            batch_size=checkpoint["batch_size"],
            device=device,
        )
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.history = checkpoint["history"]
        return trainer
