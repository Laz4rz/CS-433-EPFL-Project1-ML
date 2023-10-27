import numpy as np
import helpers as hp
import implementations as impl

print("Loading data...")
x_train, x_test, y_train, train_ids, test_ids = hp.load_csv_data("data/")

y_train = np.expand_dims(y_train, 1)
y_train = y_train.reshape((y_train.shape[0],1))

import src.utils.constants as c
import src.features.build_features as bf

x_train_nonans, removed_cols = bf.build_train_features(data=x_train, percentage=c.PERCENTAGE_NAN)

lambda_ = 0.1                                      # regularization parameter
max_iters = 10                                     # max number of iterations 
threshold = 1e-8                                   # threshold for stopping criterion
gamma = 0.4                                        # step size
initial_w = np.zeros((x_train_nonans.shape[1], 1)) # initial weights

print("Running GD...")
w_mean_squared_error_gd, loss_mean_squared_error_gd = impl.mean_squared_error_gd(y_train, x_train_nonans, initial_w, max_iters, gamma)

print("Mean Squared Error Gradient Descent: ", loss_mean_squared_error_gd)
