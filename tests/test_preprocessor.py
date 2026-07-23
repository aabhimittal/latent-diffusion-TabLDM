import numpy as np
import pandas as pd

from tabular_ldm.data import TabularPreprocessor


class TestTabularPreprocessor:
    def test_autodetect_columns(self, mixed_df):
        pp = TabularPreprocessor()
        pp.fit(mixed_df)
        assert set(pp.numerical_cols) == {"age", "income", "score"}
        assert set(pp.categorical_cols) == {"gender", "category"}

    def test_feature_dim(self, mixed_df):
        pp = TabularPreprocessor()
        pp.fit(mixed_df)
        # 3 numerical + 3 (gender) + 4 (category)
        assert pp.feature_dim == 3 + 3 + 4

    def test_transform_shape(self, mixed_df):
        pp = TabularPreprocessor()
        X = pp.fit_transform(mixed_df)
        assert X.shape == (len(mixed_df), pp.feature_dim)
        assert X.dtype == np.float32

    def test_numerical_standardized(self, mixed_df):
        pp = TabularPreprocessor()
        X = pp.fit_transform(mixed_df)
        # Numerical slice should be roughly zero-mean, unit-variance
        num_slice = X[:, : len(pp.numerical_cols)]
        assert abs(num_slice.mean()) < 0.1
        assert abs(num_slice.std() - 1.0) < 0.2

    def test_categorical_one_hot(self, mixed_df):
        pp = TabularPreprocessor()
        X = pp.fit_transform(mixed_df)
        slices = pp.get_cat_slices()
        for col, (start, end) in slices.items():
            block = X[:, start:end]
            # Each row is a valid one-hot vector
            assert np.allclose(block.sum(axis=1), 1.0)
            assert ((block == 0) | (block == 1)).all()

    def test_inverse_roundtrip_numerical(self, numerical_df):
        pp = TabularPreprocessor()
        X = pp.fit_transform(numerical_df)
        reconstructed = pp.inverse_transform(X)
        for col in numerical_df.columns:
            np.testing.assert_allclose(
                reconstructed[col].values,
                numerical_df[col].values,
                atol=1e-4,
            )

    def test_inverse_roundtrip_categorical(self, mixed_df):
        pp = TabularPreprocessor()
        X = pp.fit_transform(mixed_df)
        reconstructed = pp.inverse_transform(X)
        for col in ["gender", "category"]:
            assert (reconstructed[col].values == mixed_df[col].values).all()

    def test_explicit_columns(self, mixed_df):
        pp = TabularPreprocessor(numerical_cols=["age"], categorical_cols=["gender"])
        X = pp.fit_transform(mixed_df)
        assert X.shape[1] == 1 + 3  # 1 numerical + 3 gender categories

    def test_fit_transform_equals_fit_then_transform(self, mixed_df):
        pp1 = TabularPreprocessor()
        X1 = pp1.fit_transform(mixed_df)

        pp2 = TabularPreprocessor()
        pp2.fit(mixed_df)
        X2 = pp2.transform(mixed_df)

        np.testing.assert_array_equal(X1, X2)

    def test_purely_numerical(self, numerical_df):
        pp = TabularPreprocessor()
        X = pp.fit_transform(numerical_df)
        assert X.shape == (150, 3)
        assert pp.categorical_cols == []


class TestConstraints:
    def test_integer_column_detected(self, mixed_df):
        pp = TabularPreprocessor()
        pp.fit(mixed_df)
        # 'age' is integer-valued (stored as float), income/score are continuous.
        assert "age" in pp.int_cols
        assert "income" not in pp.int_cols

    def test_ranges_recorded(self, mixed_df):
        pp = TabularPreprocessor()
        pp.fit(mixed_df)
        lo, hi = pp.num_ranges["age"]
        assert lo == mixed_df["age"].min()
        assert hi == mixed_df["age"].max()

    def test_integer_column_rounded_on_inverse(self):
        df = pd.DataFrame({"count": [1.0, 2.0, 3.0, 4.0, 5.0] * 10})
        pp = TabularPreprocessor()
        pp.fit(df)
        # Feed a non-integer standardized value; inverse must round it.
        X = pp.transform(df)
        X[:, 0] += 0.13  # perturb within-range
        out = pp.inverse_transform(X)
        assert np.issubdtype(out["count"].dtype, np.integer)

    def test_values_clamped_to_range(self):
        df = pd.DataFrame({"x": np.linspace(0.0, 10.0, 50)})
        pp = TabularPreprocessor()
        X = pp.fit_transform(df)
        # Push some encoded values far outside the trained range.
        X[:, 0] += 100.0
        out = pp.inverse_transform(X)
        assert out["x"].max() <= 10.0 + 1e-6
        assert out["x"].min() >= 0.0 - 1e-6

    def test_constraints_can_be_disabled(self):
        df = pd.DataFrame({"x": np.linspace(0.0, 10.0, 50)})
        pp = TabularPreprocessor(enforce_constraints=False)
        X = pp.fit_transform(df)
        X[:, 0] += 100.0
        out = pp.inverse_transform(X)
        assert out["x"].max() > 10.0  # not clamped
