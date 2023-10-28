#!/usr/bin/env python
# encoding: utf-8
"""
parameters.py: File containing parameters dataclass.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class Parameters:
    seed: int = 42
    lambda_: float = 0.1
    iters: iters = 10
    gamma: float = 0.01
    threshold = 1e-8
    batch_size: int = 32
    initial_w: np.ndarray
