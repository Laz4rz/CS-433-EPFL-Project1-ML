#!/usr/bin/env python
# encoding: utf-8
"""
build_data.py: File containing the code used for data preprocessing.
"""

import numpy as np
import src.utils.constants as c


def standardize(data: np.ndarray) -> np.ndarray:
    """Standardize the dataset.

    Args:
        data: the dataset to be standardized.
    Returns:
        new_data: the standardized dataset.
    """

    new_data = np.c_[np.ones((data.shape[0], 1)), data]

    mean_x = np.mean(new_data)
    new_data = new_data - mean_x
    std_x = np.std(new_data)
    new_data = new_data / std_x

    return new_data


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


def less_than_percent_nans(
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
    nan_percentage_per_column = np.isnan(x).sum(0) / len(x)
    less_than_percent_nans_columns_mask = nan_percentage_per_column < (percentage / 100)
    removed_columns = np.arange(x.shape[1])[~less_than_percent_nans_columns_mask]
    return x[:, less_than_percent_nans_columns_mask], removed_columns


def build_train_features(
    data: np.ndarray, percentage: int = c.PERCENTAGE_NAN
) -> np.ndarray:
    """Build the train features.

    Args:
        data (np.ndarray): train data.
        percentage (float, optional): Percentage of NaN values in columns to be removed. Defaults to 90.

    Returns:
        np.ndarray: the train features.
        np.ndarray: indexes of the columns with more than percentage NaN values.
    """
    x_train_standardized, removed_cols = less_than_percent_nans(
        x=data, percentage=percentage
    )
    x_train_standardized = replace_nan_mean(x=x_train_standardized)
    x_train_standardized = standardize(data=x_train_standardized)
    return x_train_standardized, removed_cols


def build_test_features(
    x_test: np.ndarray, removed_cols: np.ndarray = []
) -> np.ndarray:
    """Build the test features.

    Args:
        x_test (np.ndarray):. test data
        removed_cols (np.ndarray): columns to be removed.

    Returns:
        np.ndarray: the test features.
    """

    cols = np.arange(x_test.shape[1])
    cols = np.delete(cols, removed_cols)
    x_test_standardized = x_test[:, cols]
    x_test_standardized = replace_nan_mean(x=x_test_standardized)
    x_test_standardized = standardize(data=x_test_standardized)
    return x_test_standardized
