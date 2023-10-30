#!/usr/bin/env python
# encoding: utf-8
"""
parameters.py: File containing parameters dataclass.
"""

from dataclasses import dataclass


@dataclass
class Parameters:
    seed: int = 42  # Random seed
    lambda_: float = 0.1  # Regularization parameter
    iters: int = 10  # Number of iterations
    gamma: float = 0.01  # Learning rate
    batch_size: int = 32  # Batch size
    degree: int = 1  # Degree of the polynomial
    balance: bool = True  # Balance the dataset
    balance_scale: int = 3  # Scale of the balancing
    drop_calculated: bool = True  # Drop the calculated features
    percentage: int = 90  # Percentage of NaNs to use for dropping.
    fill_nans: str = "random"  # How to fill the NaNs
    how_init: str = "random"  # How to initialize the weights
    drop_outliers: bool = None  # Remove outliers
