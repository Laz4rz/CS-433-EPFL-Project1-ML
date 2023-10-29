#!/usr/bin/env python
# encoding: utf-8
"""
build_data.py: File containing the code used for data preprocessing.
"""

import re as re
import os as os
import numpy as np
import src.utils.constants as c

from typing import Tuple


def drop_calculated_features(x: np.ndarray) -> np.ndarray:
    """Drop calculated features.
       These features are the ones that start with an underscore
       (Exceptions:  _DENSTR2, _GEOSTR, and _STATE).

    Args:
        x (np.ndarray): dataset.

    Returns:
        np.ndarray: dataset without calculated features.
        np.ndarray: indexes of the calculated features.
    """
    with open(os.path.join("./data", "x_train.csv")) as f:
        first_line = f.readline().strip("\n")
    cols_name = first_line.split(",")
    cols_name = np.asarray(cols_name)
    cols_name = cols_name[1:]
    # rgx = "^_(?!DENSTR2$|GEOSTR$|STATE$)[A-Za-z_]\w*"
    # calculated_features_idx = np.where([re.match(rgx, col) for col in cols_name])[0]
    startwith_ = np.array(list(map(lambda x: x.startswith("_"), cols_name)))
    endwith_ = np.array(list(map(lambda x: x.endswith("_"), cols_name)))
    to_drop = np.logical_or(startwith_, endwith_)
    exclude = list(map(lambda x: x not in ["_DENSTR2", "_GEOSTR", "_STATE"], cols_name))
    to_drop = np.logical_and(to_drop, exclude)
    calculated_features_idx = np.where(to_drop)[0]
    new_data = np.delete(x, calculated_features_idx, axis=1)
    return new_data, calculated_features_idx


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)
    """

    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def standardize(data: np.ndarray) -> np.ndarray:
    """Standardize the dataset.

    Args:
        data: the dataset to be standardized.
    Returns:
        new_data: the standardized dataset.
    """

    # new_data = np.c_[np.ones((data.shape[0], 1)), data]
    new_data = data.copy()

    mean = np.mean(new_data, axis=0)
    std = np.std(new_data, axis=0)
    std = np.where(std == 0, 1, std)
    return (new_data - mean) / std


def replace_nan_mean(x: np.ndarray) -> np.ndarray:
    """Replace NaN values with the mean of the column.

    Args:
        x (np.ndarray): dataset.
    Returns:
        np.ndarray: dataset with NaN values replaced with the mean of the column.
    """

    x = x.copy()
    for col in range(x.shape[1]):
        mean = np.nanmean(x[:, col])
        x[np.isnan(x[:, col]), col] = mean
    return x


def less_than_percent_nans_columns(
    x: np.ndarray, percentage: int = c.PERCENTAGE_NAN
) -> np.ndarray:
    """Remove columns with more than percentage NaN values.

    Args:
        x (np.ndarray): dataset.
        percentage (int): percentage of NaN values.

    Returns:
        np.ndarray: dataset with columns with more than percentage NaN values removed.
        np.ndarray: indexes of the columns with more than percentage NaN values.
    """

    x = x.copy()
    idx_to_drop = []
    for col in range(x.shape[1]):
        nan_percentage_row = np.isnan(x[:, col]).sum() / len(x[:, col]) * 100
        if nan_percentage_row > percentage: idx_to_drop.append(col)
    return np.delete(x, idx_to_drop, 1), idx_to_drop



def less_than_percent_nans_rows(
    x: np.ndarray, percentage: int = c.PERCENTAGE_NAN
) -> np.ndarray:
    """Remove columns with more than percentage NaN values.

    Args:
        x (np.ndarray): dataset.
        percentage (int): percentage of NaN values.

    Returns:
        np.ndarray: dataset with columns with more than percentage NaN values removed.
        np.ndarray: indexes of the columns with more than percentage NaN values.
    """

    x = x.copy()
    idx_to_drop = []
    for row in range(x.shape[0]):
        nan_percentage_row = np.isnan(x[row, :]).sum() / len(x[row, :]) * 100
        if nan_percentage_row > percentage: idx_to_drop.append(row)
    return np.delete(x, idx_to_drop, 0), idx_to_drop


def remove_outliers(x: np.ndarray, threshold: int = 9) -> np.ndarray:
    mask = (np.abs(x) > threshold).any(1)
    return np.delete(x, mask, 0), mask


def get_most_freq_value(x: np.ndarray) -> np.ndarray:
    """Get the most frequent value of a 1D array.

    Args:
        x (np.ndarray): 1D array.

    Returns:
        np.ndarray: the most frequent value.
    """
    values, counts = np.unique(x[~np.isnan(x)], return_counts=True)
    return values[np.argmax(counts)]


def replace_nan_most_freq(x: np.ndarray) -> np.ndarray:
    """Replace NaN values with the most frequent value of the column.

    Args:
        x (np.ndarray): dataset.

    Returns:
        np.ndarray: dataset with NaN values replaced with the most frequent value of the column.
    """
    x = x.copy()
    for col in range(x.shape[1]):
        most_freq_val = get_most_freq_value(x[:, col])
        x[np.isnan(x[:, col]), col] = most_freq_val
    return x

def replace_nan_random(x: np.ndarray) -> np.ndarray:
    """Replace NaN values with a random value of the column.

    Args:
        x (np.ndarray): dataset.

    Returns:
        np.ndarray: dataset with NaN values replaced with a random value of the column.
    """
    x = x.copy()
    for col in range(x.shape[1]):
        # rand = np.random.choice(x_train[~np.isnan(x_train[:, 100]), 100], np.isnan(x_train[:, 100]).sum())
        x[np.isnan(x[:, col]), col] = np.random.uniform(np.nanmin(x[:, col]), np.nanmax(x[:, col]), np.isnan(x[:, col]).sum())
    return x

def balance_data(x: np.ndarray, y: np.ndarray, scale: int = 1) -> np.ndarray:
    """Balance the dataset.

    Args:
        x (np.ndarray): dataset.
        y (np.ndarray): labels.

    Returns:
        np.ndarray: balanced dataset.
        np.ndarray: balanced labels.
    """
    y1_size = np.sum(y[y == 1])
    y0_balanced_size = int(y1_size * scale)
    y0_idx = np.random.choice(np.arange(0, len(y))[(y == -1).squeeze()], y0_balanced_size)
    balanced_idx = np.concatenate([y0_idx, np.arange(0, len(y))[(y == 1).squeeze()]])
    balanced_idx = np.sort(balanced_idx)
    y_balanced = y[balanced_idx]
    x_balanced = x[balanced_idx]
    return x_balanced, y_balanced

def build_train_features(
    x: np.ndarray, y: np.ndarray, percentage: int = c.PERCENTAGE_NAN, fill_nans: str = None, balance: bool = False, balance_scale: int = 1, drop_calculated: bool = True, polynomial_expansion_degree: int = 1, drop_outliers: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the train features.

    Args:
        x (np.ndarray): dataset.
        percentage (float, optional): Threshold of NaN values in columns to be removed. Defaults to 90.
        fill_nans (str, optional): Method to fill nan values. Defaults to None. 
        balance (bool, optional): Whether to balance the dataset or not. Defaults to False.
        balance_scale (int, optional): Scale of the balancing. Defaults to 1.
        drop_calculated (bool, optional): Whether to drop calculated features or not. Defaults to True.
    Returns:
        np.ndarray: the train features.
        np.ndarray: the train labels.
        np.ndarray: indexes of the calculated features.
        np.ndarray: indexes of the columns with more than percentage NaN values.
    """
    x_train_standardized = x.copy()
    calculated_cols_idxs = []
    if drop_calculated:
        x_train_standardized, calculated_cols_idxs = drop_calculated_features(x=x)
    x_train_standardized, more_than_nan_idxs_cols = less_than_percent_nans_columns(
        x=x_train_standardized, percentage=percentage
    )
    x_train_standardized, more_than_nan_idxs_rows = less_than_percent_nans_rows(
        x=x_train_standardized, percentage=50
    )

    y = np.delete(y, more_than_nan_idxs_rows, 0)
    
    if fill_nans is not None:
        if fill_nans == "mean":
            x_train_standardized = replace_nan_mean(x=x_train_standardized)
        elif fill_nans == "most_freq":
            x_train_standardized = replace_nan_most_freq(x=x_train_standardized)
        elif fill_nans == "random":
            x_train_standardized = replace_nan_random(x=x_train_standardized)
        assert(np.sum(np.isnan(x_train_standardized)) == 0), "There are still NaN values in the dataset."

    if drop_outliers is not None:
        x_train_standardized, outliers_mask = remove_outliers(x=x_train_standardized, threshold=drop_outliers)
        y = np.delete(y, outliers_mask, 0)
        assert(x_train_standardized.shape[0] == y.shape[0]), "The number of samples and labels is not the same."
        assert(np.sum(np.isnan(x_train_standardized)) == 0), "There are still NaN values in the dataset."

    if balance:
        x_train_standardized, y = balance_data(x=x_train_standardized, y=y, scale=balance_scale)
        assert(x_train_standardized.shape[0] == y.shape[0]), "The number of samples and labels is not the same."
        assert(np.sum(np.isnan(x_train_standardized)) == 0), "There are still NaN values in the dataset."

    x_train_standardized = build_poly(x_train_standardized, degree=polynomial_expansion_degree)

    x_train_standardized = standardize(data=x_train_standardized)

    return x_train_standardized, y, calculated_cols_idxs, more_than_nan_idxs_cols


def build_test_features(
    x: np.ndarray, idx_calc_columns: np.ndarray = [], idx_nan_percent: np.ndarray = [], fill_nans: str = None, polynomial_expansion_degree: int = 1
) -> np.ndarray:
    """Build the test features.

    Args:
        x_test (np.ndarray):. test data
        removed_cols (np.ndarray): columns to be removed.

    Returns:
        np.ndarray: the test features.
    """

    x = np.delete(np.delete(x, idx_calc_columns, 1), idx_nan_percent, 1)

    if fill_nans is not None:
        if fill_nans == "mean":
            x = replace_nan_mean(x=x)
        elif fill_nans == "most_freq":
            x = replace_nan_most_freq(x=x)
        elif fill_nans == "random":
            x = replace_nan_random(x=x)
        assert(np.sum(np.isnan(x)) == 0), "There are still NaN values in the dataset."

    x = build_poly(x, degree=polynomial_expansion_degree)

    x_standardized = standardize(data=x)
    return x_standardized

