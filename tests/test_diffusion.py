import torch
import pytest

from tabular_ldm.diffusion import DDPMScheduler, DenoisingMLP


LATENT_DIM = 16
BATCH = 32
T = 100


@pytest.fixture
def scheduler():
    return DDPMScheduler(num_timesteps=T)


@pytest.fixture
def denoiser():
    return DenoisingMLP(latent_dim=LATENT_DIM, hidden_dims=[64, 64])


class TestDDPMScheduler:
    def test_q_sample_shape(self, scheduler):
        x0 = torch.randn(BATCH, LATENT_DIM)
        t = torch.randint(0, T, (BATCH,))
        xt = scheduler.q_sample(x0, t)
        assert xt.shape == x0.shape

    def test_q_sample_at_t0_is_clean(self, scheduler):
        x0 = torch.randn(BATCH, LATENT_DIM)
        t = torch.zeros(BATCH, dtype=torch.long)
        xt = scheduler.q_sample(x0, t, noise=torch.zeros_like(x0))
        # sqrt(alpha_bar_0) ≈ 1 and sqrt(1-alpha_bar_0) ≈ 0 for small beta_start
        torch.testing.assert_close(xt, x0, atol=0.02, rtol=0)

    def test_q_sample_at_tmax_is_noisy(self, scheduler):
        x0 = torch.zeros(1000, LATENT_DIM)
        t = torch.full((1000,), T - 1, dtype=torch.long)
        xt = scheduler.q_sample(x0, t)
        # At T-1 the signal should be heavily corrupted → variance near 1
        assert xt.var().item() > 0.5

    def test_betas_monotone_increasing(self, scheduler):
        assert (scheduler.betas[1:] >= scheduler.betas[:-1]).all()

    def test_alphas_cumprod_monotone_decreasing(self, scheduler):
        assert (scheduler.alphas_cumprod[1:] <= scheduler.alphas_cumprod[:-1]).all()

    def test_p_sample_step_shape(self, scheduler):
        x_t = torch.randn(BATCH, LATENT_DIM)
        pred_noise = torch.randn(BATCH, LATENT_DIM)
        x_prev = scheduler.p_sample_step(x_t, t_idx=50, pred_noise=pred_noise)
        assert x_prev.shape == x_t.shape

    def test_p_sample_step_t0_deterministic(self, scheduler):
        x_t = torch.randn(BATCH, LATENT_DIM)
        pred_noise = torch.randn(BATCH, LATENT_DIM)
        x1 = scheduler.p_sample_step(x_t, t_idx=0, pred_noise=pred_noise)
        x2 = scheduler.p_sample_step(x_t, t_idx=0, pred_noise=pred_noise)
        torch.testing.assert_close(x1, x2)

    def test_to_device(self, scheduler):
        sched = scheduler.to(torch.device("cpu"))
        assert sched.betas.device.type == "cpu"


class TestDenoisingMLP:
    def test_unconditional_forward(self, denoiser):
        z = torch.randn(BATCH, LATENT_DIM)
        t = torch.randint(0, T, (BATCH,))
        out = denoiser(z, t)
        assert out.shape == (BATCH, LATENT_DIM)

    def test_conditional_forward(self):
        denoiser = DenoisingMLP(latent_dim=LATENT_DIM, hidden_dims=[64, 64], num_classes=3)
        z = torch.randn(BATCH, LATENT_DIM)
        t = torch.randint(0, T, (BATCH,))
        y = torch.randint(0, 3, (BATCH,))
        out = denoiser(z, t, y)
        assert out.shape == (BATCH, LATENT_DIM)

    def test_conditional_without_y_uses_zeros(self):
        denoiser = DenoisingMLP(latent_dim=LATENT_DIM, hidden_dims=[64], num_classes=4)
        z = torch.randn(BATCH, LATENT_DIM)
        t = torch.randint(0, 50, (BATCH,))
        out = denoiser(z, t, y=None)
        assert out.shape == (BATCH, LATENT_DIM)

    def test_output_finite(self, denoiser):
        z = torch.randn(BATCH, LATENT_DIM)
        t = torch.randint(0, T, (BATCH,))
        out = denoiser(z, t)
        assert torch.isfinite(out).all()

    def test_gradients_flow(self, denoiser):
        z = torch.randn(BATCH, LATENT_DIM)
        t = torch.randint(0, T, (BATCH,))
        noise = torch.randn(BATCH, LATENT_DIM)
        pred = denoiser(z, t)
        loss = torch.nn.functional.mse_loss(pred, noise)
        loss.backward()
        for p in denoiser.parameters():
            assert p.grad is not None
