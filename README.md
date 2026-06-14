# TabLDM — Latent Diffusion for Tabular Data

**TabLDM** is a production-ready Python package that applies Latent Diffusion Models (LDMs) to tabular data synthesis. It is the first clean implementation that runs DDPM *in a compressed latent space* for tabular data, making it fundamentally different from TabDDPM (which diffuses in raw data space).

## Why TabLDM?

| Method  | Diffusion space | Mixed types | Class-conditional | Stable training |
|---------|----------------|-------------|-------------------|-----------------|
| CTGAN   | —              | ✓           | ✗                 | ✗ (GAN)        |
| TabDDPM | Raw data       | ✓           | ✓                 | ✓               |
| **TabLDM** | **Latent**  | **✓**       | **✓**             | **✓**           |

The latent bottleneck means the diffusion model works on a smaller, smoother manifold — improving sample quality and enabling faster inference.

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

### Class-conditional generation

```python
import numpy as np

# Generate 500 fraud + 500 legit rows
labels = np.array([0] * 500 + [1] * 500)
synthetic = model.generate(1000, labels=labels)
```

### Evaluate fidelity

```python
from tabular_ldm.metrics import column_shapes, ml_efficacy, privacy_distance

shapes   = column_shapes(real_df, synthetic)
efficacy = ml_efficacy(real_df, synthetic, target_col="label")
privacy  = privacy_distance(real_df, synthetic)

print(f"TSTR/TRTR ratio: {efficacy['efficacy_ratio']:.3f}")
print(f"NNDR (privacy):  {privacy['nndr_mean']:.3f}")
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
