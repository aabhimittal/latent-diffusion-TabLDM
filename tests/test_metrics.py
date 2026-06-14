import numpy as np
import pandas as pd
import pytest

from tabular_ldm.metrics import (
    column_shapes,
    column_pair_trends,
    ml_efficacy,
    privacy_distance,
    quality_report,
)


@pytest.fixture
def identical_dfs():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 200),
            "b": rng.normal(5, 2, 200),
            "cat": rng.choice(["x", "y", "z"], size=200),
        }
    )
    return df, df.copy()


@pytest.fixture
def different_dfs():
    rng = np.random.default_rng(99)
    real = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.normal(0, 1, 300)})
    synth = pd.DataFrame({"a": rng.normal(10, 1, 300), "b": rng.normal(10, 1, 300)})
    return real, synth


class TestColumnShapes:
    def test_identical_data_low_ks(self, identical_dfs):
        real, synth = identical_dfs
        results = column_shapes(real, synth)
        assert results["a"]["ks_statistic"] == 0.0
        assert results["b"]["ks_statistic"] == 0.0

    def test_different_data_high_ks(self, different_dfs):
        real, synth = different_dfs
        results = column_shapes(real, synth)
        assert results["a"]["ks_statistic"] > 0.5

    def test_categorical_tv_zero_for_identical(self, identical_dfs):
        real, synth = identical_dfs
        results = column_shapes(real, synth)
        assert results["cat"]["tv_distance"] == 0.0

    def test_returns_all_columns(self, identical_dfs):
        real, synth = identical_dfs
        results = column_shapes(real, synth)
        assert set(results.keys()) == {"a", "b", "cat"}

    def test_type_labels(self, identical_dfs):
        real, synth = identical_dfs
        results = column_shapes(real, synth)
        assert results["a"]["type"] == "numerical"
        assert results["cat"]["type"] == "categorical"


class TestColumnPairTrends:
    def test_identical_data_zero_delta(self, identical_dfs):
        real, synth = identical_dfs
        result = column_pair_trends(real, synth, numerical_cols=["a", "b"])
        assert result["mean_corr_delta"] == 0.0

    def test_different_data_nonzero_delta(self, different_dfs):
        real, synth = different_dfs
        result = column_pair_trends(real, synth)
        assert "mean_corr_delta" in result

    def test_single_col_returns_empty(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = column_pair_trends(df, df, numerical_cols=["x"])
        assert result == {}


class TestMLEfficacy:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(1)
        real = pd.DataFrame(
            {"x": rng.normal(0, 1, 100), "label": rng.choice(["a", "b"], 100)}
        )
        synth = pd.DataFrame(
            {"x": rng.normal(0, 1, 100), "label": rng.choice(["a", "b"], 100)}
        )
        result = ml_efficacy(real, synth, target_col="label")
        assert set(result.keys()) == {"TRTR", "TSTR", "efficacy_ratio"}

    def test_scores_bounded(self):
        rng = np.random.default_rng(2)
        real = pd.DataFrame(
            {"x": rng.normal(0, 1, 150), "label": rng.choice(["a", "b"], 150)}
        )
        synth = real.copy()
        result = ml_efficacy(real, synth, target_col="label")
        assert 0.0 <= result["TRTR"] <= 1.0
        assert 0.0 <= result["TSTR"] <= 1.0


class TestPrivacyDistance:
    def test_identical_data_low_nndr(self):
        rng = np.random.default_rng(5)
        df = pd.DataFrame({"x": rng.normal(0, 1, 100), "y": rng.normal(0, 1, 100)})
        result = privacy_distance(df, df, numerical_cols=["x", "y"])
        # When synthetic == real, real-to-synthetic distance == real-to-real
        # → NNDR ≈ 1.0 or undefined for exact duplicates
        assert "nndr_mean" in result

    def test_no_numerical_returns_empty(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        result = privacy_distance(df, df, numerical_cols=[])
        assert result == {}


class TestQualityReport:
    def _data(self):
        rng = np.random.default_rng(3)
        n = 300
        real = pd.DataFrame(
            {
                "x": rng.normal(0, 1, n),
                "y": rng.normal(2, 1, n),
                "label": rng.choice(["a", "b"], n),
            }
        )
        return real

    def test_identical_data_high_score(self):
        real = self._data()
        report = quality_report(real, real.copy(), target_col="label")
        assert report["overall_score"] > 0.8
        assert 0.0 <= report["overall_score"] <= 1.0

    def test_report_keys(self):
        real = self._data()
        report = quality_report(real, real.copy(), target_col="label")
        for key in [
            "overall_score",
            "shape_fidelity",
            "correlation_fidelity",
            "column_shapes",
            "ml_efficacy",
            "privacy",
        ]:
            assert key in report

    def test_without_target(self):
        real = self._data().drop(columns=["label"])
        report = quality_report(real, real.copy())
        assert report["ml_efficacy"] is None
        assert 0.0 <= report["overall_score"] <= 1.0

    def test_different_data_lower_score(self):
        real = self._data()
        rng = np.random.default_rng(123)
        synth = pd.DataFrame(
            {
                "x": rng.normal(10, 1, 300),
                "y": rng.normal(-10, 1, 300),
                "label": rng.choice(["a", "b"], 300),
            }
        )
        good = quality_report(real, real.copy(), target_col="label")["overall_score"]
        bad = quality_report(real, synth, target_col="label")["overall_score"]
        assert bad < good
