import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Optional, Tuple


class TabularPreprocessor:
    """Encodes mixed-type tabular data into a flat float32 vector.

    Numerical columns are standardized; categorical columns are label-encoded
    then one-hot encoded. Inverse transform reconstructs a DataFrame.
    """

    def __init__(
        self,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        enforce_constraints: bool = True,
    ):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.enforce_constraints = enforce_constraints
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.cat_dims: Dict[str, int] = {}
        self.num_ranges: Dict[str, Tuple[float, float]] = {}
        self.int_cols: List[str] = []
        self.feature_dim: int = 0
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        if self.numerical_cols is None:
            self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.categorical_cols is None:
            self.categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in self.numerical_cols:
            scaler = StandardScaler()
            scaler.fit(df[[col]])
            self.scalers[col] = scaler
            # Record the observed range and whether the column is integer-valued,
            # so generated samples can be clamped/rounded back to realistic values.
            col_vals = df[col].to_numpy(dtype=np.float64)
            self.num_ranges[col] = (float(np.nanmin(col_vals)), float(np.nanmax(col_vals)))
            finite = col_vals[np.isfinite(col_vals)]
            is_int = pd.api.types.is_integer_dtype(df[col]) or (
                finite.size > 0 and np.allclose(finite, np.round(finite))
            )
            if is_int:
                self.int_cols.append(col)

        for col in self.categorical_cols:
            enc = LabelEncoder()
            enc.fit(df[col].astype(str))
            self.encoders[col] = enc
            self.cat_dims[col] = len(enc.classes_)

        self.feature_dim = len(self.numerical_cols) + sum(self.cat_dims.values())
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        if self.numerical_cols:
            num = np.concatenate(
                [self.scalers[c].transform(df[[c]]) for c in self.numerical_cols],
                axis=1,
            )
            parts.append(num)
        for col in self.categorical_cols:
            labels = self.encoders[col].transform(df[col].astype(str))
            one_hot = np.eye(self.cat_dims[col], dtype=np.float32)[labels]
            parts.append(one_hot)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        """Reconstruct a DataFrame from a float32 encoded matrix."""
        idx = 0
        result: Dict[str, np.ndarray] = {}

        # getattr guards keep checkpoints saved before constraints were added loadable.
        enforce = getattr(self, "enforce_constraints", False)
        num_ranges = getattr(self, "num_ranges", {})
        int_cols = getattr(self, "int_cols", [])

        for col in self.numerical_cols:
            val = self.scalers[col].inverse_transform(X[:, idx : idx + 1]).ravel()
            if enforce and col in num_ranges:
                lo, hi = num_ranges[col]
                val = np.clip(val, lo, hi)
            if enforce and col in int_cols:
                val = np.rint(val).astype(np.int64)
            result[col] = val
            idx += 1

        for col in self.categorical_cols:
            n = self.cat_dims[col]
            one_hot = X[:, idx : idx + n]
            labels = np.argmax(one_hot, axis=1)
            result[col] = self.encoders[col].inverse_transform(labels)
            idx += n

        col_order = list(self.numerical_cols) + list(self.categorical_cols)
        return pd.DataFrame(result)[col_order]

    def get_num_slice(self) -> Tuple[int, int]:
        """Return (start, end) indices for the numerical portion of the vector."""
        return 0, len(self.numerical_cols)

    def get_cat_slices(self) -> Dict[str, Tuple[int, int]]:
        """Return {col: (start, end)} for each categorical column's one-hot block."""
        slices = {}
        idx = len(self.numerical_cols)
        for col in self.categorical_cols:
            n = self.cat_dims[col]
            slices[col] = (idx, idx + n)
            idx += n
        return slices
