"""Example: Synthetic Adult Income dataset generation.

Demonstrates class-conditional synthesis on the classic Adult dataset,
useful for privacy-preserving data sharing in HR/finance.

Run:  python examples/adult_income.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from tabular_ldm import TabularLDM
from tabular_ldm.metrics import column_shapes, ml_efficacy


def main():
    print("Fetching Adult Income dataset...")
    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame.dropna().reset_index(drop=True)

    # Keep a manageable subset
    df = df.sample(5000, random_state=42).reset_index(drop=True)
    target_col = "class"
    print(f"Dataset: {df.shape} | Label distribution:\n{df[target_col].value_counts()}\n")

    model = TabularLDM(
        latent_dim=24,
        vae_hidden_dims=[256, 128],
        diffusion_hidden_dims=[512, 512, 256],
        num_timesteps=500,
        kl_weight=0.05,
    )
    model.fit(df, target_col=target_col, vae_epochs=80, diffusion_epochs=200, verbose=True)

    print("\nGenerating 2000 synthetic rows (balanced 50/50)...")
    n = 2000
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    synth = model.generate(n, labels=labels)
    synth[target_col] = model._label_encoder.inverse_transform(labels)

    feature_cols = [c for c in df.columns if c != target_col]
    shapes = column_shapes(df[feature_cols], synth[feature_cols])
    num_ks = [v["ks_statistic"] for v in shapes.values() if v["type"] == "numerical"]
    cat_tv = [v["tv_distance"] for v in shapes.values() if v["type"] == "categorical"]

    print(f"\nMean KS (numerical): {np.mean(num_ks):.3f}")
    print(f"Mean TV (categorical): {np.mean(cat_tv):.3f}")

    eff = ml_efficacy(df, synth, target_col=target_col)
    print(f"\nTRTR: {eff['TRTR']:.3f}  TSTR: {eff['TSTR']:.3f}  Ratio: {eff['efficacy_ratio']:.3f}")

    model.save("adult_income_model/")
    print("\nModel saved to adult_income_model/")


if __name__ == "__main__":
    main()
