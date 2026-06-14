import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mixed_df():
    """Small DataFrame with both numerical and categorical columns."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=n).astype(float),
            "income": rng.normal(50_000, 15_000, size=n),
            "score": rng.uniform(0, 1, size=n),
            "gender": rng.choice(["M", "F", "Other"], size=n),
            "category": rng.choice(["A", "B", "C", "D"], size=n),
        }
    )


@pytest.fixture
def numerical_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x": rng.normal(0, 1, size=150),
            "y": rng.exponential(2, size=150),
            "z": rng.uniform(-1, 1, size=150),
        }
    )
