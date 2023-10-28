import numpy as np
import helpers as hp
import implementations as impl
import src.utils.constants as c
import src.features.build_features as bf

print("Loading data...")
x_train, x_test, y_train, train_ids, test_ids = hp.load_csv_data("data/")

print(x_train.dtype.name)

y_train = np.expand_dims(y_train, 1)
y_train = y_train.reshape((y_train.shape[0], 1))

print(x_train.shape)
print("Building features...")
x_train_nonans, removed_cols = bf.build_train_features(
    x=x_train, percentage=c.PERCENTAGE_NAN, fill_nans="mean"
)
print(x_train_nonans.shape)

# lambda_ = 0.1
# max_iters = 10
# threshold = 1e-8
# gamma = 0.4
# initial_w = np.zeros((x_train_nonans.shape[1], 1))

# print("Running gradient descent...")
# w_mean_squared_error_gd, loss_mean_squared_error_gd = impl.mean_squared_error_gd(y_train, x_train_nonans, initial_w, max_iters, gamma)

# print("Loss:{loss} ".format(loss=loss_mean_squared_error_gd))
