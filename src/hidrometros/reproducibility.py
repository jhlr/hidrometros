"""Seeding helpers. Original scaffold by Mikael de Albuquerque Mulatinho."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (CPU, CUDA and MPS) so runs are repeatable.

    MPS kernels are not bit-exact deterministic; this bounds, but does not remove, run-to-run noise.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding derived from the torch initial seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
