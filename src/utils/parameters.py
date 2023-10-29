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
    iters: int = 10
    gamma: float = 0.01
    batch_size: int = 32
    degree: int = 1
    balance: bool = True  # Added missing type annotation
    balance_scale: int = 3
    drop_calculated: bool = True  # Added missing type annotation
    percentage: int = 90
    fill_nans: str = "random"
    how_init: str = "random"
