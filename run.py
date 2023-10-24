import numpy as np
from utils import *
from implementations import *
from helpers import *


def replace_nan_mean(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    for col in range(x.shape[1]):
        mean = np.nanmean(x[:, col])
        x[np.isnan(x[:, col]), col] = mean
    return x


def less_than_percent_nans(x: np.ndarray, percentage: int) -> np.ndarray:
    x = x.copy()
    nan_percentage_per_column = np.isnan(x).sum(0) / len(x)
    less_than_percent_nans_columns_mask = nan_percentage_per_column < (percentage / 100)
    return x[:, less_than_percent_nans_columns_mask]


# data load
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/")

# data preparation
y_train = np.expand_dims(y_train, 1)
print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)
print("x_test shape: ", x_test.shape)

# clear nans
x_train_nonans = replace_nan_mean(less_than_percent_nans(x_train, 90))

# standardize
x_train_nonans = standardize(x_train_nonans)[:, :3]

# parameters
max_iters = 20  # max number of iterations
threshold = 1e-8  # threshold for stopping criterion
gamma = 0.4  # step size
initial_w = np.zeros((x_train_nonans.shape[1], 1))  # initial weights

# Mean squared error gradient descent
w_mean_squared_error_gd, loss_mean_squared_error_gd = mean_squared_error_gd(
    y_train, x_train_nonans, initial_w, max_iters, gamma
)  # why constant first loss 0.04415103539701647 ?

rmse_tr = np.sqrt(2 * loss_mean_squared_error_gd)
# rmse_te = np.sqrt(2 * compute_loss(y_test, x_test, w_mean_squared_error_gd))

print(
    "Mean squared error gradient descent: W: {w}, Loss:{loss}".format(
        w=w_mean_squared_error_gd, loss=loss_mean_squared_error_gd
    )
)
# print("RMSE train: {rmse_tr}, RMSE test: {rmse_te}".format(rmse_tr=rmse_tr, rmse_te=rmse_te))
