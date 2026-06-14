"""TabularLDM: Latent Diffusion Model for tabular data synthesis."""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..data.preprocessor import TabularPreprocessor
from ..diffusion.network import DenoisingMLP
from ..diffusion.scheduler import DDPMScheduler
from ..vae.tabular_vae import TabularVAE


class TabularLDM:
    """Two-stage latent diffusion model for tabular data.

    Stage 1 – VAE: learn a compressed continuous latent space from the raw
    tabular features (numerical + categorical mixed types).

    Stage 2 – DDPM: train a denoising network on the *latent* representations
    produced by the VAE encoder. Generation runs DDPM sampling in latent space
    and decodes back to tabular rows via the VAE decoder.

    This is fundamentally different from TabDDPM, which diffuses in raw data
    space. The latent bottleneck:
    - Removes the need to handle mixed types inside the diffusion model.
    - Lets the diffusion model work on a much smaller, smoother manifold.
    - Enables class-conditional generation via label embeddings.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        vae_hidden_dims: Optional[List[int]] = None,
        diffusion_hidden_dims: Optional[List[int]] = None,
        num_timesteps: int = 1000,
        kl_weight: float = 0.05,
        num_classes: int = 0,
        device: str = "auto",
    ):
        self.latent_dim = latent_dim
        self.vae_hidden_dims = vae_hidden_dims or [256, 128]
        self.diffusion_hidden_dims = diffusion_hidden_dims or [512, 512, 256, 256]
        self.num_timesteps = num_timesteps
        self.kl_weight = kl_weight
        self.num_classes = num_classes

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.preprocessor: Optional[TabularPreprocessor] = None
        self.vae: Optional[TabularVAE] = None
        self.scheduler: Optional[DDPMScheduler] = None
        self.denoiser: Optional[DenoisingMLP] = None
        self._input_dim: int = 0
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _init_models(self, input_dim: int) -> None:
        self._input_dim = input_dim
        self.vae = TabularVAE(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.vae_hidden_dims,
            kl_weight=self.kl_weight,
        ).to(self.device)

        self.scheduler = DDPMScheduler(num_timesteps=self.num_timesteps).to(self.device)

        self.denoiser = DenoisingMLP(
            latent_dim=self.latent_dim,
            hidden_dims=self.diffusion_hidden_dims,
            num_classes=self.num_classes,
        ).to(self.device)

    # ------------------------------------------------------------------
    # Stage 1: train the VAE
    # ------------------------------------------------------------------

    def fit_vae(
        self,
        X: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> List[float]:
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        optimizer = optim.Adam(self.vae.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        losses: List[float] = []
        self.vae.train()
        for epoch in range(epochs):
            total = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon, mu, logvar = self.vae(batch)
                loss = self.vae.loss(batch, recon, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
            scheduler.step()
            avg = total / len(loader)
            losses.append(avg)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  VAE [{epoch+1:>4d}/{epochs}] loss={avg:.5f}")
        return losses

    # ------------------------------------------------------------------
    # Stage 2: train the diffusion model on latent codes
    # ------------------------------------------------------------------

    def fit_diffusion(
        self,
        X: np.ndarray,
        epochs: int = 300,
        batch_size: int = 256,
        lr: float = 2e-4,
        labels: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> List[float]:
        self.vae.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            mu, _ = self.vae.encode(X_t)
            Z = mu.cpu()

        tensors = [Z]
        if labels is not None:
            tensors.append(torch.tensor(labels, dtype=torch.long))
        dataset = TensorDataset(*tensors)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        optimizer = optim.AdamW(self.denoiser.parameters(), lr=lr, weight_decay=1e-4)
        lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        losses: List[float] = []
        self.denoiser.train()
        for epoch in range(epochs):
            total = 0.0
            for batch in loader:
                z0 = batch[0].to(self.device)
                y = batch[1].to(self.device) if labels is not None else None

                t = torch.randint(0, self.scheduler.T, (z0.size(0),), device=self.device)
                noise = torch.randn_like(z0)
                z_noisy = self.scheduler.q_sample(z0, t, noise)

                pred_noise = self.denoiser(z_noisy, t, y)
                loss = nn.functional.mse_loss(pred_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
                optimizer.step()
                total += loss.item()

            lr_sched.step()
            avg = total / len(loader)
            losses.append(avg)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Diffusion [{epoch+1:>4d}/{epochs}] loss={avg:.5f}")
        return losses

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        vae_epochs: int = 100,
        diffusion_epochs: int = 300,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> "TabularLDM":
        """Fit both VAE and diffusion stages on a DataFrame.

        Args:
            df: Input DataFrame with mixed numerical/categorical columns.
            numerical_cols: Columns to treat as numerical. Auto-detected if None.
            categorical_cols: Columns to treat as categorical. Auto-detected if None.
            target_col: Optional label column for conditional generation.
                        Must be set if num_classes > 0.
            vae_epochs: Training epochs for Stage 1.
            diffusion_epochs: Training epochs for Stage 2.
        """
        feature_df = df.drop(columns=[target_col]) if target_col else df
        self.preprocessor = TabularPreprocessor(numerical_cols, categorical_cols)
        X = self.preprocessor.fit_transform(feature_df)

        labels: Optional[np.ndarray] = None
        if target_col is not None:
            from sklearn.preprocessing import LabelEncoder

            self._label_encoder = LabelEncoder()
            labels = self._label_encoder.fit_transform(df[target_col].astype(str))
            if self.num_classes == 0:
                self.num_classes = len(self._label_encoder.classes_)

        self._init_models(X.shape[1])

        if verbose:
            print(
                f"[TabLDM] Stage 1 — VAE  "
                f"(input={X.shape[1]}, latent={self.latent_dim}, "
                f"device={self.device})"
            )
        self.fit_vae(X, epochs=vae_epochs, batch_size=batch_size, lr=lr, verbose=verbose)

        if verbose:
            print(f"\n[TabLDM] Stage 2 — Diffusion (T={self.num_timesteps})")
        self.fit_diffusion(
            X,
            epochs=diffusion_epochs,
            batch_size=batch_size,
            lr=lr / 5,
            labels=labels,
            verbose=verbose,
        )

        self._fitted = True
        return self

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        labels: Optional[np.ndarray] = None,
        guidance_scale: float = 1.0,
    ) -> pd.DataFrame:
        """Sample n_samples rows from the learned distribution.

        Args:
            n_samples: Number of synthetic rows to generate.
            labels: Integer class labels for conditional generation.
                    Length must equal n_samples. Ignored if num_classes == 0.
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance).
                            Values > 1.0 strengthen the class conditioning signal.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before generate().")

        self.vae.eval()
        self.denoiser.eval()

        z = torch.randn(n_samples, self.latent_dim, device=self.device)

        y: Optional[torch.Tensor] = None
        if labels is not None and self.num_classes > 0:
            y = torch.tensor(labels, dtype=torch.long, device=self.device)

        use_cfg = guidance_scale > 1.0 and y is not None
        uncond_y = torch.zeros(n_samples, dtype=torch.long, device=self.device) if use_cfg else None

        for t_idx in reversed(range(self.scheduler.T)):
            t = torch.full((n_samples,), t_idx, device=self.device, dtype=torch.long)
            pred_noise = self.denoiser(z, t, y)

            if use_cfg:
                uncond_noise = self.denoiser(z, t, uncond_y)
                pred_noise = uncond_noise + guidance_scale * (pred_noise - uncond_noise)

            z = self.scheduler.p_sample_step(z, t_idx, pred_noise)

        X_recon = self.vae.decode(z).cpu().numpy()
        return self.preprocessor.inverse_transform(X_recon)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and config to directory `path`."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.vae.state_dict(), os.path.join(path, "vae.pt"))
        torch.save(self.denoiser.state_dict(), os.path.join(path, "denoiser.pt"))
        with open(os.path.join(path, "preprocessor.pkl"), "wb") as f:
            pickle.dump(self.preprocessor, f)
        meta: Dict = {
            "latent_dim": self.latent_dim,
            "vae_hidden_dims": self.vae_hidden_dims,
            "diffusion_hidden_dims": self.diffusion_hidden_dims,
            "num_timesteps": self.num_timesteps,
            "kl_weight": self.kl_weight,
            "num_classes": self.num_classes,
            "input_dim": self._input_dim,
        }
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        if hasattr(self, "_label_encoder"):
            with open(os.path.join(path, "label_encoder.pkl"), "wb") as f:
                pickle.dump(self._label_encoder, f)

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "TabularLDM":
        """Load a saved TabularLDM from directory `path`."""
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        with open(os.path.join(path, "preprocessor.pkl"), "rb") as f:
            preprocessor = pickle.load(f)

        model = cls(
            latent_dim=meta["latent_dim"],
            vae_hidden_dims=meta["vae_hidden_dims"],
            diffusion_hidden_dims=meta["diffusion_hidden_dims"],
            num_timesteps=meta["num_timesteps"],
            kl_weight=meta["kl_weight"],
            num_classes=meta["num_classes"],
            device=device,
        )
        model._init_models(meta["input_dim"])
        model.preprocessor = preprocessor

        model.vae.load_state_dict(
            torch.load(os.path.join(path, "vae.pt"), map_location=model.device, weights_only=True)
        )
        model.denoiser.load_state_dict(
            torch.load(os.path.join(path, "denoiser.pt"), map_location=model.device, weights_only=True)
        )

        le_path = os.path.join(path, "label_encoder.pkl")
        if os.path.exists(le_path):
            with open(le_path, "rb") as f:
                model._label_encoder = pickle.load(f)

        model._fitted = True
        return model
