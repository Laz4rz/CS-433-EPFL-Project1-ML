#!/usr/bin/env python
# encoding: utf-8
"""
parameters.py: File containing parameters dataclass.
"""

import src.utils.constants as c

from dataclasses import dataclass


@dataclass
class Parameters:
    seed: int = 42  # Random seed
    lambda_: float = 0.1  # Regularization parameter
    iters: int = 10  # Number of iterations
    gamma: float = 0.15  # Learning rate
    batch_size: int = 32  # Batch size
    degree: int = 1  # Degree of the polynomial
    balance: bool = False  # Balance the dataset
    balance_scale: int = 1  # Scale of the balancing
    drop_calculated: bool = False  # Drop the calculated features
    percentage_col: int = (
        c.PERCENTAGE_NAN
    )  # Percentage of NaNs to use for dropping columns.
    percentage_row: int = (
        c.PERCENTAGE_NAN
    )  # Percentage of NaNs to use for dropping rows.
    fill_nans: str = (
        "with_num"  # How to fill the NaNs (with_num, with_mean, with_num, random)
    )
    how_init: str = "zeros"  # How to initialize the weights (unos, zeros, random)
    drop_outliers: bool = None  # Remove outliers
    num: int = 0  # Replacer for the NaNs (if fill_nans == "with_num")
