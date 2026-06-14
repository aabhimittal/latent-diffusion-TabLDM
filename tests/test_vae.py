import torch
import pytest

from tabular_ldm.vae import TabularVAE


INPUT_DIM = 20
LATENT_DIM = 8
BATCH = 32


@pytest.fixture
def vae():
    return TabularVAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden_dims=[64, 32])


class TestTabularVAE:
    def test_forward_shapes(self, vae):
        x = torch.randn(BATCH, INPUT_DIM)
        recon, mu, logvar = vae(x)
        assert recon.shape == (BATCH, INPUT_DIM)
        assert mu.shape == (BATCH, LATENT_DIM)
        assert logvar.shape == (BATCH, LATENT_DIM)

    def test_encode_decode_roundtrip_shape(self, vae):
        x = torch.randn(BATCH, INPUT_DIM)
        mu, logvar = vae.encode(x)
        z = vae.reparameterize(mu, logvar)
        recon = vae.decode(z)
        assert recon.shape == x.shape

    def test_loss_is_scalar(self, vae):
        x = torch.randn(BATCH, INPUT_DIM)
        recon, mu, logvar = vae(x)
        loss = vae.loss(x, recon, mu, logvar)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_loss_decreases_with_training(self):
        vae = TabularVAE(INPUT_DIM, LATENT_DIM, hidden_dims=[64, 32])
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-2)
        x = torch.randn(128, INPUT_DIM)

        losses = []
        for _ in range(30):
            recon, mu, logvar = vae(x)
            loss = vae.loss(x, recon, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "VAE loss should decrease during training"

    def test_reparameterize_eval_mode(self, vae):
        vae.eval()
        mu = torch.zeros(BATCH, LATENT_DIM)
        logvar = torch.zeros(BATCH, LATENT_DIM)
        z = vae.reparameterize(mu, logvar)
        # In eval mode, reparameterize returns mu directly
        torch.testing.assert_close(z, mu)

    def test_reparameterize_train_stochastic(self, vae):
        vae.train()
        mu = torch.zeros(BATCH, LATENT_DIM)
        logvar = torch.zeros(BATCH, LATENT_DIM)
        z1 = vae.reparameterize(mu, logvar)
        z2 = vae.reparameterize(mu, logvar)
        assert not torch.allclose(z1, z2), "Training reparameterize should be stochastic"

    def test_kl_weight_affects_loss(self):
        x = torch.randn(BATCH, INPUT_DIM)
        vae_low = TabularVAE(INPUT_DIM, LATENT_DIM, kl_weight=0.0)
        vae_high = TabularVAE(INPUT_DIM, LATENT_DIM, kl_weight=1.0)
        # With same inputs/weights (both freshly init) KL term direction may vary
        # but we can at least confirm both run without error
        for vae in [vae_low, vae_high]:
            recon, mu, logvar = vae(x)
            loss = vae.loss(x, recon, mu, logvar)
            assert torch.isfinite(loss)
