# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**TabLDM** — a Python package implementing Latent Diffusion Models for tabular data synthesis.
The key architectural idea: run DDPM in a *compressed VAE latent space* rather than raw data space.
This separates the mixed-type encoding problem (VAE's job) from the generative model (DDPM's job).

## Commands

```bash
# Install
pip install -e ".[dev]"

# All tests
pytest tests/

# Single test module
pytest tests/test_vae.py

# Single test
pytest tests/test_integration.py::TestEndToEnd::test_save_and_load

# Demo (trains a model end-to-end on synthetic fraud data)
python demo.py

# Real-world example (downloads Adult Income dataset via sklearn)
python examples/adult_income.py
```

## Architecture

Two-stage pipeline: `TabularPreprocessor → TabularVAE → DDPMScheduler + DenoisingMLP`

### Stage 1: Preprocessing + VAE

`tabular_ldm/data/preprocessor.py` — `TabularPreprocessor`
- Auto-detects numerical vs categorical columns.
- Numerical: `StandardScaler`. Categorical: label-encode then one-hot.
- `inverse_transform()` reconstructs a DataFrame from float32 vectors.
- `get_cat_slices()` returns `{col: (start, end)}` for one-hot blocks — used in tests.

`tabular_ldm/vae/tabular_vae.py` — `TabularVAE`
- Standard β-VAE: `TabularEncoder` → (μ, log σ²), reparameterize, `TabularDecoder`.
- `reparameterize` is stochastic in train mode, deterministic (returns μ) in eval mode.
- Loss = MSE reconstruction + `kl_weight` × KL divergence.
- Hidden dims are MirrorED: encoder goes `[256, 128]`, decoder goes `[128, 256]`.

### Stage 2: Diffusion in latent space

`tabular_ldm/diffusion/scheduler.py` — `DDPMScheduler`
- Linear β schedule from `beta_start` to `beta_end` over `T` steps.
- `q_sample(x0, t, noise)` — forward process.
- `p_sample_step(x_t, t_idx, pred_noise)` — single reverse step using the simplified DDPM formula: `mean = (x_t - β/√(1-ᾱ) · ε) / √α`. Adds noise for `t > 0`.
- Call `.to(device)` to move all precomputed tensors to GPU.

`tabular_ldm/diffusion/network.py` — `DenoisingMLP`
- Input: latent vector + sinusoidal time embedding + optional class embedding.
- Body: `input_proj → [ResidualBlock] × N`. Each `ResidualBlock` gets the conditioning vector added before the residual path.
- Class conditioning uses `nn.Embedding(num_classes + 1, time_embed_dim)` — index 0 is the unconditional token (used for CFG null conditioning).

### Orchestration

`tabular_ldm/models/tabular_ldm.py` — `TabularLDM`
- `fit()`: calls `fit_vae()` then `fit_diffusion()`. Accepts a `target_col` for class-conditional mode.
- `fit_diffusion()`: encodes the full dataset to latent μ (eval mode, no reparameterization) before training starts. This makes diffusion training deterministic w.r.t. the VAE.
- `generate()`: runs the full DDPM reverse loop then decodes. Supports classifier-free guidance via `guidance_scale > 1.0`.
- `save()`/`load()`: pickles preprocessor + meta dict; saves model weights as `.pt` files.

### Metrics

`tabular_ldm/metrics/statistical.py`
- `column_shapes` — KS test (numerical) + Total Variation distance (categorical).
- `column_pair_trends` — mean absolute Pearson correlation delta across column pairs.
- `ml_efficacy` — TRTR and TSTR accuracy using `RandomForestClassifier`.
- `privacy_distance` — Nearest-Neighbour Distance Ratio (NNDR); values ≥ 1 indicate low memorisation.

## Key design decisions

- **Latent encoding at diffusion train time uses μ only** (not a reparameterized sample). This reduces variance in the diffusion training targets.
- **kl_weight=0.05** by default — small enough that the posterior doesn't collapse but tight enough for DDPM to learn the prior.
- **Scheduler lives on the same device as the model** — always call `scheduler.to(device)` inside `_init_models()`.
- **`num_classes` + 1 embedding** — index 0 is the unconditional embedding slot for CFG; user-provided labels start from 0 but are offset internally if using CFG.

## Test layout

| File | What it covers |
|------|---------------|
| `test_preprocessor.py` | Roundtrip, shape, one-hot validity, auto-detection |
| `test_vae.py` | Forward shapes, loss, reparameterization modes |
| `test_diffusion.py` | Scheduler math, denoiser shapes/gradients |
| `test_metrics.py` | All four metric functions |
| `test_integration.py` | Full pipeline: fit → generate → save → load |
