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
- Supports `schedule="linear"` or `schedule="cosine"` (Nichol & Dhariwal). Cosine is the default chosen by `TabularLDM`.
- Precomputes `snr` (signal-to-noise per step) for Min-SNR loss weighting.
- `q_sample(x0, t, noise)` — forward process.
- `p_sample_step(x_t, t_idx, pred_noise)` — ancestral DDPM reverse step: `mean = (x_t - β/√(1-ᾱ) · ε) / √α`. Adds noise for `t > 0`.
- `ddim_step(x_t, t_idx, t_prev_idx, pred_noise, eta=0)` — deterministic DDIM reverse step (Song et al.), enabling sampling on a sparse timestep subsequence.
- `inference_timesteps(n)` — evenly spaced descending subsequence for DDIM.
- Call `.to(device)` to move all precomputed tensors to GPU.

`tabular_ldm/diffusion/network.py` — `DenoisingMLP`
- Input: latent vector + sinusoidal time embedding + optional class embedding.
- Body: `input_proj → [ResidualBlock] × N`. Each `ResidualBlock` gets the conditioning vector added before the residual path.
- Class conditioning uses `nn.Embedding(num_classes + 1, time_embed_dim)` — index 0 is the unconditional token. The network treats `y` as an *already-offset* index where 0 means "null"; the orchestrator passes `label + 1` for real classes (see below).

`tabular_ldm/diffusion/ema.py` — `EMA`
- Exponential moving average of denoiser weights. Updated every optimizer step during diffusion training; `copy_to()` is applied at the end so generation samples from the smoothed weights.

### Orchestration

`tabular_ldm/models/tabular_ldm.py` — `TabularLDM`
- `fit()`: calls `fit_vae()` then `fit_diffusion()`. Accepts a `target_col` for class-conditional mode.
- `fit_diffusion()`: encodes the full dataset to latent μ (eval mode, no reparameterization) before training starts. This makes diffusion training deterministic w.r.t. the VAE. Applies CFG label dropout (`cfg_dropout`), Min-SNR-γ loss weighting (`min_snr_gamma`), and EMA.
- `_diffusion_loss()`: ε-prediction MSE, optionally reweighted by `min(SNR, γ)/SNR` (Min-SNR-γ, Hang et al. 2023). Set `min_snr_gamma=None` for plain MSE.
- `generate()`: full ancestral DDPM by default; pass `num_inference_steps` for fast DDIM. Classifier-free guidance via `guidance_scale > 1.0`.
- `save()`/`load()`: pickles preprocessor + meta dict (now includes `schedule`, `cfg_dropout`, `min_snr_gamma`, `ema_decay`, `target_col`); saves model weights as `.pt` files. `load()` uses `meta.get(...)` defaults so older checkpoints still load.

### Command-line interface

`tabular_ldm/cli.py` (entry point `python -m tabular_ldm`) — three subcommands:
- `fit <csv> --target <col> --out <dir>` — train and save.
- `generate <model_dir> --n <N> --out <csv> [--label L --guidance G --steps S]` — sample; `--steps` triggers DDIM.
- `evaluate <real.csv> <synth.csv> --target <col>` — prints a `quality_report` as JSON.

### Metrics

`tabular_ldm/metrics/statistical.py`
- `column_shapes` — KS test (numerical) + Total Variation distance (categorical).
- `column_pair_trends` — mean absolute Pearson correlation delta across column pairs.
- `ml_efficacy` — TRTR and TSTR accuracy using `RandomForestClassifier`.
- `privacy_distance` — Nearest-Neighbour Distance Ratio (NNDR); values ≥ 1 indicate low memorisation.
- `quality_report` — single-call scorecard combining the above into an `overall_score` ∈ [0, 1]. Gracefully skips ML efficacy if `target_col` is absent from either frame.

## Key design decisions

- **Latent encoding at diffusion train time uses μ only** (not a reparameterized sample). This reduces variance in the diffusion training targets.
- **kl_weight=0.05** by default — small enough that the posterior doesn't collapse but tight enough for DDPM to learn the prior.
- **Scheduler lives on the same device as the model** — always call `scheduler.to(device)` inside `_init_models()`.
- **Label offset for CFG** — `nn.Embedding(num_classes + 1, …)` reserves index 0 as the null token. The orchestrator passes `label + 1` for real classes in both `fit_diffusion` and `generate`, and `0` for the unconditional pass. This is essential: without the offset, class 0 collides with the null token and guidance silently breaks. Training must also *see* the null token, which is why `cfg_dropout > 0` randomly maps labels to 0 during training.
- **EMA before sampling** — diffusion training ends with `ema.copy_to(denoiser)`; never sample from the raw last-step weights.
- **Min-SNR weighting is on by default** (`min_snr_gamma=5.0`); set to `None` to disable.

## Test layout

| File | What it covers |
|------|---------------|
| `test_preprocessor.py` | Roundtrip, shape, one-hot validity, auto-detection |
| `test_vae.py` | Forward shapes, loss, reparameterization modes |
| `test_diffusion.py` | Scheduler math (linear + cosine), DDIM, EMA, denoiser shapes/gradients |
| `test_metrics.py` | All metric functions incl. `quality_report` |
| `test_integration.py` | Full pipeline: fit → generate (DDPM + DDIM + guidance) → save → load |
| `test_cli.py` | CLI fit → generate → evaluate roundtrip |
