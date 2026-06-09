"""Random seed helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def set_random_seed(seed: int) -> None:
    """Set Python and NumPy random seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
