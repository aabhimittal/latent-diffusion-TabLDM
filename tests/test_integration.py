"""End-to-end integration tests.

These train for very few steps (fast), so they verify the pipeline assembles
and runs without error rather than checking generation quality.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from tabular_ldm import TabularLDM


def make_dataset(n=300, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=n).astype(float),
            "income": rng.normal(50_000, 10_000, size=n),
            "region": rng.choice(["north", "south", "east", "west"], size=n),
            "label": rng.choice(["churn", "retain"], size=n),
        }
    )


@pytest.fixture
def fast_model():
    return TabularLDM(
        latent_dim=8,
        vae_hidden_dims=[32, 16],
        diffusion_hidden_dims=[64, 64],
        num_timesteps=50,
        device="cpu",
    )


class TestEndToEnd:
    def test_fit_generates_correct_shape(self, fast_model):
        df = make_dataset()
        fast_model.fit(df, target_col="label", vae_epochs=3, diffusion_epochs=5, verbose=False)
        synth = fast_model.generate(n_samples=50)
        assert len(synth) == 50
        assert set(synth.columns) == {"age", "income", "region"}

    def test_generated_dtypes_match(self, fast_model):
        df = make_dataset()
        fast_model.fit(df, target_col="label", vae_epochs=3, diffusion_epochs=5, verbose=False)
        synth = fast_model.generate(100)
        assert pd.api.types.is_numeric_dtype(synth["age"])
        assert pd.api.types.is_numeric_dtype(synth["income"])
        assert pd.api.types.is_string_dtype(synth["region"])

    def test_conditional_generation(self, fast_model):
        df = make_dataset()
        fast_model.fit(df, target_col="label", vae_epochs=3, diffusion_epochs=5, verbose=False)
        labels = np.zeros(80, dtype=int)
        synth = fast_model.generate(80, labels=labels)
        assert len(synth) == 80

    def test_no_target_col(self):
        model = TabularLDM(
            latent_dim=8,
            vae_hidden_dims=[32],
            diffusion_hidden_dims=[64],
            num_timesteps=20,
            device="cpu",
        )
        df = make_dataset()[["age", "income", "region"]]
        model.fit(df, vae_epochs=3, diffusion_epochs=3, verbose=False)
        synth = model.generate(30)
        assert len(synth) == 30

    def test_save_and_load(self, fast_model):
        df = make_dataset()
        fast_model.fit(df, target_col="label", vae_epochs=3, diffusion_epochs=3, verbose=False)
        synth_before = fast_model.generate(20)

        with tempfile.TemporaryDirectory() as tmpdir:
            fast_model.save(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "vae.pt"))
            assert os.path.exists(os.path.join(tmpdir, "denoiser.pt"))

            loaded = TabularLDM.load(tmpdir, device="cpu")
            synth_after = loaded.generate(20)

        assert list(synth_before.columns) == list(synth_after.columns)
        assert len(synth_after) == 20

    def test_generate_before_fit_raises(self, fast_model):
        with pytest.raises(RuntimeError, match="fit()"):
            fast_model.generate(10)

    def test_numerical_only_dataset(self):
        model = TabularLDM(
            latent_dim=4,
            vae_hidden_dims=[16],
            diffusion_hidden_dims=[32],
            num_timesteps=10,
            device="cpu",
        )
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.uniform(0, 5, 200)})
        model.fit(df, vae_epochs=3, diffusion_epochs=3, verbose=False)
        synth = model.generate(50)
        assert synth.shape == (50, 2)

    def test_vae_loss_history_returned(self, fast_model):
        df = make_dataset()
        fast_model._init_models(5)
        X = np.random.randn(100, 5).astype(np.float32)
        losses = fast_model.fit_vae(X, epochs=5, verbose=False)
        assert len(losses) == 5
        assert all(isinstance(v, float) for v in losses)

    def test_ddim_fast_sampling(self, fast_model):
        df = make_dataset()
        fast_model.fit(df, target_col="label", vae_epochs=3, diffusion_epochs=5, verbose=False)
        synth = fast_model.generate(40, num_inference_steps=10)
        assert len(synth) == 40
        assert set(synth.columns) == {"age", "income", "region"}

    def test_guidance_runs(self, fast_model):
        df = make_dataset()
        fast_model.fit(df, target_col="label", vae_epochs=3, diffusion_epochs=5, verbose=False)
        labels = np.zeros(30, dtype=int)
        synth = fast_model.generate(30, labels=labels, guidance_scale=2.0)
        assert len(synth) == 30

    def test_guidance_with_ddim(self, fast_model):
        df = make_dataset()
        fast_model.fit(df, target_col="label", vae_epochs=3, diffusion_epochs=5, verbose=False)
        labels = np.ones(25, dtype=int)
        synth = fast_model.generate(
            25, labels=labels, guidance_scale=3.0, num_inference_steps=8
        )
        assert len(synth) == 25

    def test_cosine_schedule_end_to_end(self):
        model = TabularLDM(
            latent_dim=8,
            vae_hidden_dims=[32],
            diffusion_hidden_dims=[64],
            num_timesteps=30,
            schedule="cosine",
            device="cpu",
        )
        df = make_dataset()
        model.fit(df, target_col="label", vae_epochs=3, diffusion_epochs=4, verbose=False)
        assert model.scheduler.schedule == "cosine"
        synth = model.generate(20, num_inference_steps=10)
        assert len(synth) == 20

    def test_config_survives_save_load(self, fast_model):
        df = make_dataset()
        fast_model.fit(df, target_col="label", vae_epochs=2, diffusion_epochs=2, verbose=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            fast_model.save(tmpdir)
            loaded = TabularLDM.load(tmpdir, device="cpu")
        assert loaded.schedule == fast_model.schedule
        assert loaded.min_snr_gamma == fast_model.min_snr_gamma
        assert loaded._target_col == "label"

    def test_min_snr_none_runs(self):
        model = TabularLDM(
            latent_dim=6,
            vae_hidden_dims=[16],
            diffusion_hidden_dims=[32],
            num_timesteps=20,
            min_snr_gamma=None,
            device="cpu",
        )
        df = make_dataset()[["age", "income", "region"]]
        model.fit(df, vae_epochs=2, diffusion_epochs=3, verbose=False)
        assert len(model.generate(10)) == 10
