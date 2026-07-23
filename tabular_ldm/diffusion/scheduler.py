import math
from typing import List

import torch


def _linear_betas(T: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


def _cosine_betas(T: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal, "Improved DDPM" (2021).

    Defines ᾱ_t = f(t)/f(0) with f(t) = cos((t/T + s)/(1+s) · π/2)², then
    recovers β_t = 1 − ᾱ_t/ᾱ_{t-1}. Spends more steps at low noise, which
    improves likelihood and sample quality versus the linear schedule.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(max=max_beta).float()


class DDPMScheduler:
    """DDPM noise scheduler supporting linear and cosine schedules.

    Precomputes all derived quantities (cumulative products, posterior variance,
    SNR) once at construction. All tensors live on CPU and are moved to the
    model's device via ``.to(device)``.

    Sampling supports both the ancestral DDPM reverse process (``p_sample_step``)
    and the deterministic DDIM process (``ddim_step``), the latter enabling fast
    sampling with far fewer than ``T`` steps.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        self.T = num_timesteps
        self.schedule = schedule

        if schedule == "linear":
            betas = _linear_betas(num_timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = _cosine_betas(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule '{schedule}' (use 'linear' or 'cosine')")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
        # Clamp to avoid log(0) at t=0 where alpha_bar_prev == 1
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp(min=1e-8)
        )
        # Signal-to-noise ratio per timestep, used for Min-SNR loss weighting.
        self.snr = alphas_cumprod / (1.0 - alphas_cumprod).clamp(min=1e-8)

    def to(self, device: torch.device) -> "DDPMScheduler":
        for attr in [
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
            "snr",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0) = N(sqrt(ᾱ_t) x_0, (1-ᾱ_t) I)."""
        if noise is None:
            noise = torch.randn_like(x0)
        s_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        s_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return s_alpha * x0 + s_one_minus * noise

    def get_velocity(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """v-prediction target v = sqrt(ᾱ_t)·ε − sqrt(1-ᾱ_t)·x0.

        The velocity parameterization (Salimans & Ho, 2022) is numerically
        better-behaved than ε-prediction near t=0 and pairs well with the
        cosine schedule.
        """
        s_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        s_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return s_alpha * noise - s_one_minus * x0

    def velocity_to_noise(
        self,
        x_t: torch.Tensor,
        t_idx: int,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a predicted velocity to the equivalent ε at scalar step t_idx.

        ε = sqrt(1-ᾱ_t)·x_t + sqrt(ᾱ_t)·v  (see Salimans & Ho, 2022, Appendix D).
        This lets the DDPM/DDIM reverse steps stay ε-based regardless of the
        network's prediction target.
        """
        s_alpha = self.sqrt_alphas_cumprod[t_idx]
        s_one_minus = self.sqrt_one_minus_alphas_cumprod[t_idx]
        return s_one_minus * x_t + s_alpha * v


    def p_sample_step(
        self,
        x_t: torch.Tensor,
        t_idx: int,
        pred_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Single ancestral reverse step: x_{t-1} ~ p_θ(x_{t-1} | x_t).

        Uses the simplified DDPM denoising formula from Ho et al. (2020).
        """
        alpha = self.alphas[t_idx]
        beta = self.betas[t_idx]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_idx]

        mean = (x_t - beta / sqrt_one_minus * pred_noise) / alpha.sqrt()

        if t_idx > 0:
            var = self.posterior_variance[t_idx]
            mean = mean + var.sqrt() * torch.randn_like(x_t)
        return mean

    def ddim_step(
        self,
        x_t: torch.Tensor,
        t_idx: int,
        t_prev_idx: int,
        pred_noise: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Single DDIM reverse step (Song et al., 2021).

        With ``eta=0`` the process is deterministic, which is what enables
        sampling on a sparse subsequence of timesteps. ``t_prev_idx`` is the
        next (smaller) timestep in the subsequence, or -1 for the final step.
        """
        abar_t = self.alphas_cumprod[t_idx]
        abar_prev = (
            self.alphas_cumprod[t_prev_idx]
            if t_prev_idx >= 0
            else torch.ones_like(abar_t)
        )

        # Predict x0 from the current noise estimate.
        x0_pred = (x_t - (1 - abar_t).sqrt() * pred_noise) / abar_t.sqrt()

        # Stochasticity controlled by eta (eta=0 => deterministic DDIM).
        sigma = (
            eta
            * ((1 - abar_prev) / (1 - abar_t)).sqrt()
            * (1 - abar_t / abar_prev).sqrt()
        )
        dir_xt = (1 - abar_prev - sigma**2).clamp(min=0).sqrt() * pred_noise
        x_prev = abar_prev.sqrt() * x0_pred + dir_xt

        if eta > 0 and t_prev_idx >= 0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)
        return x_prev

    def inference_timesteps(self, num_inference_steps: int) -> List[int]:
        """Evenly spaced timestep subsequence (descending) for DDIM sampling."""
        num_inference_steps = min(num_inference_steps, self.T)
        step = self.T / num_inference_steps
        steps = [int(round(self.T - 1 - i * step)) for i in range(num_inference_steps)]
        return [s for s in steps if s >= 0]
