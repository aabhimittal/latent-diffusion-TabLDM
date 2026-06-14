"""Tests for the command-line interface (fit → generate → evaluate)."""

import os

import numpy as np
import pandas as pd
import pytest

from tabular_ldm.cli import main


@pytest.fixture
def csv_dataset(tmp_path):
    rng = np.random.default_rng(11)
    n = 250
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=n).astype(float),
            "income": rng.normal(50_000, 8_000, size=n),
            "region": rng.choice(["north", "south", "east"], size=n),
            "label": rng.choice(["yes", "no"], size=n),
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestCLI:
    def test_fit_then_generate(self, csv_dataset, tmp_path):
        model_dir = str(tmp_path / "model")
        rc = main(
            [
                "fit",
                csv_dataset,
                "--target", "label",
                "--out", model_dir,
                "--latent-dim", "8",
                "--timesteps", "20",
                "--vae-epochs", "3",
                "--diffusion-epochs", "3",
                "--device", "cpu",
            ]
        )
        assert rc == 0
        assert os.path.exists(os.path.join(model_dir, "vae.pt"))

        out_csv = str(tmp_path / "synth.csv")
        rc = main(
            [
                "generate",
                model_dir,
                "--n", "30",
                "--out", out_csv,
                "--label", "0",
                "--steps", "10",
                "--device", "cpu",
            ]
        )
        assert rc == 0
        synth = pd.read_csv(out_csv)
        assert len(synth) == 30
        assert "label" in synth.columns

    def test_evaluate(self, csv_dataset, tmp_path, capsys):
        # Generate a trivial "synthetic" copy and evaluate it.
        df = pd.read_csv(csv_dataset)
        synth_path = str(tmp_path / "synth.csv")
        df.to_csv(synth_path, index=False)

        rc = main(["evaluate", csv_dataset, synth_path, "--target", "label"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Overall score" in out
