# TabLDM — Latent Diffusion for Tabular Data

**TabLDM** is a production-ready Python package that applies Latent Diffusion Models (LDMs) to tabular data synthesis. It is the first clean implementation that runs DDPM *in a compressed latent space* for tabular data, making it fundamentally different from TabDDPM (which diffuses in raw data space).

## Why TabLDM?

| Method  | Diffusion space | Mixed types | Class-conditional | Stable training |
|---------|----------------|-------------|-------------------|-----------------|
| CTGAN   | —              | ✓           | ✗                 | ✗ (GAN)        |
| TabDDPM | Raw data       | ✓           | ✓                 | ✓               |
| **TabLDM** | **Latent**  | **✓**       | **✓**             | **✓**           |

The latent bottleneck means the diffusion model works on a smaller, smoother manifold — improving sample quality and enabling faster inference.

### Research features

TabLDM ships with the techniques that make modern diffusion models work well:

- **Cosine noise schedule** (Nichol & Dhariwal, *Improved DDPM*) — default.
- **DDIM sampling** (Song et al.) — deterministic generation in 20–50 steps instead of 1000 (≈10× faster).
- **Classifier-free guidance** (Ho & Salimans) — done correctly with a reserved null token and conditioning dropout during training, so `guidance_scale` genuinely steers class-conditional output.
- **v-prediction** (Salimans & Ho) — an alternative, more stable training target; enable with `prediction_type="v"`.
- **Min-SNR-γ loss weighting** (Hang et al. 2023) — faster, more stable diffusion convergence.
- **EMA weights** — sampling from an exponential moving average of the denoiser.

### Application features

- **`augment()`** — one call to balance imbalanced classes with synthetic minority rows.
- **Realistic outputs** — integer columns and value ranges are learned and enforced on generated data.
- **Reproducibility** — `random_state` (training) and `seed` (generation) give bit-for-bit repeatable runs.

## Industry Applications

- **Fraud detection**: Oversample rare fraud cases for imbalanced classifiers.
- **Healthcare**: Generate privacy-preserving patient records (NNDR privacy metric included).
- **Finance**: Synthetic transaction data for model development without exposing PII.
- **Data augmentation**: Expand small datasets while preserving statistical structure.

## Quick Start

```bash
pip install -e .
```

```python
import pandas as pd
from tabular_ldm import TabularLDM

df = pd.read_csv("my_data.csv")

model = TabularLDM(latent_dim=32, num_timesteps=1000)
model.fit(df, target_col="label", vae_epochs=100, diffusion_epochs=300)

synthetic = model.generate(n_samples=1000)
```

### Class-conditional generation (with guidance + fast sampling)

```python
import numpy as np

# Generate 500 fraud + 500 legit rows, steered by classifier-free guidance,
# using fast 50-step DDIM sampling.
labels = np.array([0] * 500 + [1] * 500)
synthetic = model.generate(
    1000, labels=labels, guidance_scale=2.0, num_inference_steps=50
)
```

### Balance an imbalanced dataset

```python
# Fraud is 5% of the data? Generate synthetic fraud rows until classes match.
balanced = model.augment(df, target_col="label", guidance_scale=2.0)
# balanced now has ~50/50 classes: original rows + synthetic minority rows.
```

### Reproducible runs

```python
# Deterministic training and sampling.
model = TabularLDM(latent_dim=32, random_state=42)
model.fit(df, target_col="label")
a = model.generate(1000, seed=7)
b = model.generate(1000, seed=7)   # identical to `a`
```

### Evaluate fidelity — one call

```python
from tabular_ldm.metrics import quality_report

report = quality_report(real_df, synthetic, target_col="label")
print(f"Overall score:   {report['overall_score']:.3f}")   # 0–1, higher better
print(f"Shape fidelity:  {report['shape_fidelity']:.3f}")
print(f"TSTR/TRTR ratio: {report['ml_efficacy']['efficacy_ratio']:.3f}")
print(f"NNDR (privacy):  {report['privacy']['nndr_mean']:.3f}")
```

Individual metrics (`column_shapes`, `column_pair_trends`, `ml_efficacy`, `privacy_distance`) are also available directly.

### Command line

```bash
# Train and save
python -m tabular_ldm fit data.csv --target label --out model/

# Generate (DDIM 50 steps, guidance 2.0, class 1)
python -m tabular_ldm generate model/ --n 1000 --out synth.csv \
    --label 1 --guidance 2.0 --steps 50

# Score synthetic vs real
python -m tabular_ldm evaluate data.csv synth.csv --target label

# Balance an imbalanced dataset
python -m tabular_ldm augment model/ data.csv --target label --out balanced.csv --steps 50
```

### Save & load

```python
model.save("my_model/")
model = TabularLDM.load("my_model/")
```

## Run the demo

```bash
python demo.py
```

## Architecture

```
Input DataFrame
      │
      ▼
TabularPreprocessor        ← StandardScaler (numerical) + one-hot (categorical)
      │
      ▼
TabularVAE (Stage 1)       ← Encoder → μ, σ  |  Decoder
      │                       MLP with LayerNorm + GELU
      │  latent z ∈ ℝ^d
      ▼
DDPMScheduler              ← linear β schedule, q_sample, p_sample_step
      │
      ▼
DenoisingMLP (Stage 2)     ← ResidualMLP + sinusoidal time embedding
                              + optional class embedding (CFG-ready)
```

Two-stage training:
1. Train VAE to compress tabular rows into a Gaussian latent space.
2. Freeze VAE; encode training data to latent codes; train DDPM on those codes.
3. At generation: sample from DDPM, decode with VAE decoder.

## Install for development

```bash
pip install -e ".[dev]"
pytest tests/                 # all 50 tests
pytest tests/test_vae.py      # single module
python demo.py                # end-to-end demo
```
