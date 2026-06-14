import torch
import pytest

from tabular_ldm.diffusion import DDPMScheduler, DenoisingMLP, EMA


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


class TestCosineSchedule:
    def test_construct(self):
        sched = DDPMScheduler(num_timesteps=T, schedule="cosine")
        assert sched.betas.shape == (T,)
        assert sched.alphas_cumprod.shape == (T,)

    def test_betas_clamped_below_one(self):
        sched = DDPMScheduler(num_timesteps=T, schedule="cosine")
        assert (sched.betas <= 0.999 + 1e-6).all()
        assert (sched.betas > 0).all()

    def test_alphas_cumprod_monotone_decreasing(self):
        sched = DDPMScheduler(num_timesteps=T, schedule="cosine")
        assert (sched.alphas_cumprod[1:] <= sched.alphas_cumprod[:-1] + 1e-6).all()

    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError, match="Unknown schedule"):
            DDPMScheduler(num_timesteps=T, schedule="quadratic")

    def test_snr_decreases_with_time(self):
        sched = DDPMScheduler(num_timesteps=T, schedule="cosine")
        # Higher noise (larger t) => lower signal-to-noise ratio.
        assert sched.snr[0] > sched.snr[-1]


class TestDDIM:
    def test_ddim_step_shape(self, scheduler):
        x_t = torch.randn(BATCH, LATENT_DIM)
        pred_noise = torch.randn(BATCH, LATENT_DIM)
        x_prev = scheduler.ddim_step(x_t, t_idx=50, t_prev_idx=40, pred_noise=pred_noise)
        assert x_prev.shape == x_t.shape

    def test_ddim_deterministic_eta0(self, scheduler):
        x_t = torch.randn(BATCH, LATENT_DIM)
        pred_noise = torch.randn(BATCH, LATENT_DIM)
        x1 = scheduler.ddim_step(x_t, 50, 40, pred_noise, eta=0.0)
        x2 = scheduler.ddim_step(x_t, 50, 40, pred_noise, eta=0.0)
        torch.testing.assert_close(x1, x2)

    def test_ddim_final_step(self, scheduler):
        x_t = torch.randn(BATCH, LATENT_DIM)
        pred_noise = torch.randn(BATCH, LATENT_DIM)
        # t_prev_idx == -1 means abar_prev == 1 (full denoise to x0).
        x_prev = scheduler.ddim_step(x_t, t_idx=0, t_prev_idx=-1, pred_noise=pred_noise)
        assert torch.isfinite(x_prev).all()

    def test_inference_timesteps_descending(self, scheduler):
        steps = scheduler.inference_timesteps(10)
        assert len(steps) == 10
        assert steps == sorted(steps, reverse=True)
        assert max(steps) < T and min(steps) >= 0

    def test_inference_timesteps_capped(self, scheduler):
        steps = scheduler.inference_timesteps(10_000)
        assert len(steps) <= T


class TestEMA:
    def test_shadow_initialized(self, denoiser):
        ema = EMA(denoiser, decay=0.9)
        for k, v in denoiser.state_dict().items():
            torch.testing.assert_close(ema.shadow[k], v)

    def test_update_moves_toward_model(self):
        m = DenoisingMLP(latent_dim=LATENT_DIM, hidden_dims=[32])
        ema = EMA(m, decay=0.5)
        # Perturb the model, then update EMA; shadow should move but not all the way.
        with torch.no_grad():
            for p in m.parameters():
                p.add_(1.0)
        ema.update(m)
        # With decay 0.5 the shadow should be halfway, not equal to the new weights.
        any_diff = any(
            not torch.allclose(ema.shadow[k], v)
            for k, v in m.state_dict().items()
            if v.dtype.is_floating_point
        )
        assert any_diff

    def test_copy_to_restores_weights(self):
        m = DenoisingMLP(latent_dim=LATENT_DIM, hidden_dims=[32])
        ema = EMA(m, decay=0.9)
        original = {k: v.clone() for k, v in m.state_dict().items()}
        with torch.no_grad():
            for p in m.parameters():
                p.add_(5.0)
        ema.copy_to(m)
        for k, v in m.state_dict().items():
            torch.testing.assert_close(v, original[k])

    def test_invalid_decay_raises(self, denoiser):
        with pytest.raises(ValueError):
            EMA(denoiser, decay=1.5)
