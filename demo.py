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
from tabular_ldm.metrics import quality_report


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
    # 3. Fast DDIM sampling vs full DDPM
    # ------------------------------------------------------------------ #
    print_section("3. Sampling speed: full DDPM vs DDIM")
    n_gen = 1000
    labels = np.array([0] * (n_gen // 2) + [1] * (n_gen // 2))

    t0 = time.time()
    _ = model.generate(n_gen, labels=labels)
    t_full = time.time() - t0

    t0 = time.time()
    synth_df = model.generate(n_gen, labels=labels, num_inference_steps=50)
    t_fast = time.time() - t0

    print(f"\n  Full DDPM ({model.num_timesteps} steps): {t_full:.2f}s")
    print(f"  DDIM (50 steps):          {t_fast:.2f}s   →  {t_full / t_fast:.1f}x faster")

    le = model._label_encoder
    synth_df["label"] = le.inverse_transform(labels)
    print(f"  Generated shape: {synth_df.shape}")

    # ------------------------------------------------------------------ #
    # 4. Classifier-free guidance separates the classes
    # ------------------------------------------------------------------ #
    print_section("4. Classifier-free guidance (class conditioning)")
    fraud_rate_real = (df["label"] == "1").mean()
    for scale in [1.0, 3.0]:
        fraud = model.generate(
            500, labels=np.ones(500, int), guidance_scale=scale, num_inference_steps=50
        )
        # Compare a representative feature's mean under the "fraud" condition.
        print(f"  guidance={scale:>3.1f} → mean(feature_0 | fraud) = {fraud['feature_0'].mean():+.3f}")
    print(f"  (real fraud rate in training data: {fraud_rate_real:.1%})")

    # ------------------------------------------------------------------ #
    # 5. One-call quality scorecard
    # ------------------------------------------------------------------ #
    print_section("5. Quality report (single call)")
    report = quality_report(df, synth_df, target_col="label")
    print(f"\n  Overall score:        {report['overall_score']:.3f}  (0–1, higher better)")
    print(f"  Shape fidelity:       {report['shape_fidelity']:.3f}")
    print(f"  Correlation fidelity: {report['correlation_fidelity']:.3f}")
    eff = report["ml_efficacy"]
    print(f"  ML efficacy (TSTR/TRTR): {eff['efficacy_ratio']:.3f}"
          f"  (TRTR={eff['TRTR']:.3f}, TSTR={eff['TSTR']:.3f})")
    print(f"  Privacy NNDR (mean):  {report['privacy']['nndr_mean']:.3f}"
          f"  (≥1.0 ⇒ low memorisation)")

    print_section("Done")
    print("\n  Save:  model.save('my_model/')")
    print("  Load:  TabularLDM.load('my_model/')")
    print("  CLI:   python -m tabular_ldm fit data.csv --target label --out model/\n")


if __name__ == "__main__":
    main()
