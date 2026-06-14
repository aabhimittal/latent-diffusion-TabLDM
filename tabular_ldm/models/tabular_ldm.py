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
from ..diffusion.ema import EMA
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
        schedule: str = "cosine",
        cfg_dropout: float = 0.1,
        min_snr_gamma: Optional[float] = 5.0,
        ema_decay: float = 0.999,
        device: str = "auto",
    ):
        self.latent_dim = latent_dim
        self.vae_hidden_dims = vae_hidden_dims or [256, 128]
        self.diffusion_hidden_dims = diffusion_hidden_dims or [512, 512, 256, 256]
        self.num_timesteps = num_timesteps
        self.kl_weight = kl_weight
        self.num_classes = num_classes
        self.schedule = schedule
        self.cfg_dropout = cfg_dropout
        self.min_snr_gamma = min_snr_gamma
        self.ema_decay = ema_decay

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.preprocessor: Optional[TabularPreprocessor] = None
        self.vae: Optional[TabularVAE] = None
        self.scheduler: Optional[DDPMScheduler] = None
        self.denoiser: Optional[DenoisingMLP] = None
        self._input_dim: int = 0
        self._target_col: Optional[str] = None
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

        self.scheduler = DDPMScheduler(
            num_timesteps=self.num_timesteps, schedule=self.schedule
        ).to(self.device)

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
            # Offset labels by +1 so embedding index 0 stays reserved for the
            # unconditional ("null") token used by classifier-free guidance.
            tensors.append(torch.tensor(labels, dtype=torch.long) + 1)
        dataset = TensorDataset(*tensors)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        optimizer = optim.AdamW(self.denoiser.parameters(), lr=lr, weight_decay=1e-4)
        lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        ema = EMA(self.denoiser, decay=self.ema_decay)

        losses: List[float] = []
        self.denoiser.train()
        for epoch in range(epochs):
            total = 0.0
            for batch in loader:
                z0 = batch[0].to(self.device)
                y = None
                if labels is not None:
                    y = batch[1].to(self.device)
                    # Classifier-free guidance: randomly drop the label to the
                    # null token so the model learns both conditional and
                    # unconditional scores (Ho & Salimans, 2022).
                    if self.cfg_dropout > 0:
                        drop = torch.rand(y.size(0), device=self.device) < self.cfg_dropout
                        y = y.masked_fill(drop, 0)

                t = torch.randint(0, self.scheduler.T, (z0.size(0),), device=self.device)
                noise = torch.randn_like(z0)
                z_noisy = self.scheduler.q_sample(z0, t, noise)

                pred_noise = self.denoiser(z_noisy, t, y)
                loss = self._diffusion_loss(pred_noise, noise, t)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
                optimizer.step()
                ema.update(self.denoiser)
                total += loss.item()

            lr_sched.step()
            avg = total / len(loader)
            losses.append(avg)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Diffusion [{epoch+1:>4d}/{epochs}] loss={avg:.5f}")

        # Sample from the smoothed EMA weights, not the last training step.
        ema.copy_to(self.denoiser)
        return losses

    def _diffusion_loss(
        self,
        pred_noise: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """ε-prediction MSE, optionally reweighted by Min-SNR-γ.

        Min-SNR (Hang et al., 2023) caps the per-timestep loss weight at γ,
        preventing low-noise steps from dominating the gradient and speeding
        up convergence.
        """
        per_sample = nn.functional.mse_loss(pred_noise, noise, reduction="none").mean(dim=1)
        if self.min_snr_gamma is None:
            return per_sample.mean()
        snr = self.scheduler.snr[t]
        weight = torch.clamp(snr, max=self.min_snr_gamma) / snr
        return (weight * per_sample).mean()

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
        self._target_col = target_col
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
        num_inference_steps: Optional[int] = None,
    ) -> pd.DataFrame:
        """Sample n_samples rows from the learned distribution.

        Args:
            n_samples: Number of synthetic rows to generate.
            labels: Integer class labels for conditional generation.
                    Length must equal n_samples. Ignored if num_classes == 0.
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance).
                            Values > 1.0 strengthen the class conditioning signal.
            num_inference_steps: If set, use deterministic DDIM sampling over
                    this many steps (much faster). If None, run the full
                    ``num_timesteps`` ancestral DDPM reverse process.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before generate().")

        self.vae.eval()
        self.denoiser.eval()

        z = torch.randn(n_samples, self.latent_dim, device=self.device)

        y: Optional[torch.Tensor] = None
        if labels is not None and self.num_classes > 0:
            # +1 offset mirrors training; index 0 is the null token.
            y = torch.tensor(labels, dtype=torch.long, device=self.device) + 1

        use_cfg = guidance_scale > 1.0 and y is not None
        uncond_y = (
            torch.zeros(n_samples, dtype=torch.long, device=self.device) if use_cfg else None
        )

        def predict(z_t, t):
            pred = self.denoiser(z_t, t, y)
            if use_cfg:
                uncond = self.denoiser(z_t, t, uncond_y)
                pred = uncond + guidance_scale * (pred - uncond)
            return pred

        if num_inference_steps is None:
            # Full ancestral DDPM.
            for t_idx in reversed(range(self.scheduler.T)):
                t = torch.full((n_samples,), t_idx, device=self.device, dtype=torch.long)
                z = self.scheduler.p_sample_step(z, t_idx, predict(z, t))
        else:
            # Fast deterministic DDIM over a sparse subsequence.
            timesteps = self.scheduler.inference_timesteps(num_inference_steps)
            for i, t_idx in enumerate(timesteps):
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                t = torch.full((n_samples,), t_idx, device=self.device, dtype=torch.long)
                z = self.scheduler.ddim_step(z, t_idx, t_prev, predict(z, t))

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
            "schedule": self.schedule,
            "cfg_dropout": self.cfg_dropout,
            "min_snr_gamma": self.min_snr_gamma,
            "ema_decay": self.ema_decay,
            "target_col": self._target_col,
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
            schedule=meta.get("schedule", "linear"),
            cfg_dropout=meta.get("cfg_dropout", 0.1),
            min_snr_gamma=meta.get("min_snr_gamma", 5.0),
            ema_decay=meta.get("ema_decay", 0.999),
            device=device,
        )
        model._init_models(meta["input_dim"])
        model.preprocessor = preprocessor
        model._target_col = meta.get("target_col")

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
