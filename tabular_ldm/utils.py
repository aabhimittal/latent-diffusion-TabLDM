"""Utility helpers for TabLDM."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs.

    Note: full determinism on CUDA also depends on cuDNN settings and is not
    guaranteed here; this covers the sources of randomness TabLDM relies on
    (weight init, latent noise sampling, label dropout, data shuffling).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
