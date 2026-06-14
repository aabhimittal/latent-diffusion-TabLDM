"""TabLDM Demo — generates synthetic tabular data and evaluates fidelity.

Simulates a fraud-detection dataset: heavily imbalanced (5% fraud),
mixed numerical + categorical features. Demonstrates:
  1. Fitting TabLDM on real data.
  2. Generating balanced synthetic data (50/50 fraud/legit).
  3. Evaluating column-shape fidelity and ML efficacy.
"""

import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from tabular_ldm import TabularLDM
from tabular_ldm.metrics import column_shapes, ml_efficacy, privacy_distance


def make_fraud_dataset(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        weights=[0.95, 0.05],
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(6)])
    df["merchant_type"] = rng.choice(["retail", "online", "atm", "pos"], size=n)
    df["card_type"] = rng.choice(["debit", "credit", "prepaid"], size=n)
    df["label"] = y.astype(str)
    return df


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def main():
    print_section("TabLDM Demo: Synthetic Fraud Data Generation")

    # ------------------------------------------------------------------ #
    # 1. Prepare data
    # ------------------------------------------------------------------ #
    print("\n[1] Building fraud dataset (2000 rows, ~5% fraud)...")
    df = make_fraud_dataset(n=2000)
    fraud_rate = (df["label"] == "1").mean()
    print(f"    Shape: {df.shape}  |  Fraud rate: {fraud_rate:.1%}")

    # ------------------------------------------------------------------ #
    # 2. Fit TabLDM
    # ------------------------------------------------------------------ #
    print_section("2. Fitting TabLDM")
    model = TabularLDM(
        latent_dim=16,
        vae_hidden_dims=[128, 64],
        diffusion_hidden_dims=[256, 256, 128],
        num_timesteps=200,
        kl_weight=0.05,
        device="auto",
    )

    t0 = time.time()
    model.fit(
        df,
        target_col="label",
        vae_epochs=50,
        diffusion_epochs=100,
        batch_size=256,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed:.1f}s")

    # ------------------------------------------------------------------ #
    # 3. Generate balanced synthetic data
    # ------------------------------------------------------------------ #
    print_section("3. Generating 1000 balanced synthetic rows (50/50)")
    n_gen = 1000
    labels = np.array([0] * (n_gen // 2) + [1] * (n_gen // 2))
    synth_df = model.generate(n_gen, labels=labels)
    # Re-attach generated labels
    le = model._label_encoder
    synth_df["label"] = le.inverse_transform(labels)
    print(f"  Generated shape: {synth_df.shape}")
    print(f"  Columns: {list(synth_df.columns)}")

    # ------------------------------------------------------------------ #
    # 4. Evaluate fidelity
    # ------------------------------------------------------------------ #
    print_section("4. Column-shape fidelity (real vs synthetic)")
    feature_cols = [c for c in df.columns if c != "label"]
    shapes = column_shapes(df[feature_cols], synth_df[feature_cols])
    num_cols = [c for c, v in shapes.items() if v["type"] == "numerical"]
    cat_cols = [c for c, v in shapes.items() if v["type"] == "categorical"]

    print("\n  Numerical columns (KS statistic, lower is better):")
    for col in num_cols:
        ks = shapes[col]["ks_statistic"]
        bar = "█" * int(ks * 20)
        print(f"    {col:<20} KS={ks:.3f}  {bar}")

    print("\n  Categorical columns (TV distance, lower is better):")
    for col in cat_cols:
        tv = shapes[col]["tv_distance"]
        bar = "█" * int(tv * 20)
        print(f"    {col:<20} TV={tv:.3f}  {bar}")

    print_section("5. ML Efficacy (TSTR vs TRTR)")
    result = ml_efficacy(df, synth_df, target_col="label")
    print(f"\n  Train-on-Real  Test-on-Real  (TRTR): {result['TRTR']:.3f}")
    print(f"  Train-on-Synth Test-on-Real  (TSTR): {result['TSTR']:.3f}")
    ratio = result["efficacy_ratio"]
    print(f"  Efficacy ratio (TSTR/TRTR):          {ratio:.3f}  ", end="")
    if ratio >= 0.85:
        print("✓ Excellent")
    elif ratio >= 0.70:
        print("~ Acceptable")
    else:
        print("✗ Poor (train longer or tune hyperparameters)")

    print_section("6. Privacy (NNDR)")
    priv = privacy_distance(df[feature_cols], synth_df[feature_cols], numerical_cols=num_cols)
    print(f"\n  Nearest-Neighbour Distance Ratio (mean): {priv['nndr_mean']:.3f}")
    print(f"  Nearest-Neighbour Distance Ratio (median): {priv['nndr_median']:.3f}")
    print("  (Values ≥ 1.0 indicate low memorisation risk)")

    print_section("Done")
    print("\n  Model can be saved with:  model.save('my_model/')")
    print("  Reload with:             TabularLDM.load('my_model/')\n")


if __name__ == "__main__":
    main()
