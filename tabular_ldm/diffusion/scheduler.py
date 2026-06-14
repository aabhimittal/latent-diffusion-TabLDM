import torch


class DDPMScheduler:
    """Linear-beta DDPM noise scheduler.

    Precomputes all derived quantities (cumulative products, posterior variance)
    once at construction. All tensors live on CPU and are moved to the model's
    device when needed.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.T = num_timesteps

        betas = torch.linspace(beta_start, beta_end, num_timesteps)
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

    def to(self, device: torch.device) -> "DDPMScheduler":
        for attr in [
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
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

    def p_sample_step(
        self,
        x_t: torch.Tensor,
        t_idx: int,
        pred_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Single reverse step: x_{t-1} ~ p_θ(x_{t-1} | x_t).

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
