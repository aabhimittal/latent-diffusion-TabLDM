import numpy as np
import torch

from tabular_ldm import set_seed


class TestSetSeed:
    def test_torch_reproducible(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        torch.testing.assert_close(a, b)

    def test_numpy_reproducible(self):
        set_seed(7)
        a = np.random.rand(10)
        set_seed(7)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(1)
        a = torch.randn(10)
        set_seed(2)
        b = torch.randn(10)
        assert not torch.allclose(a, b)
