import numpy as np
import helpers as hp
import matplotlib.pyplot as plt
import implementations as impl
import src.utils.constants as c
import src.utils.functions as f
import src.features.build_features as bf
import src.model.predict_model as pred

from src.utils.parameters import Parameters

parameters = Parameters(
    seed=1,
    lambda_=0.1,
    iters=100,
    gamma=0.01,
)

f.set_random_seed(parameters.seed)

print("Loading data...")
x_train, x_test, y_train, train_ids, test_ids = hp.load_csv_data("data/")

print(x_train.dtype.name)

y_train = np.expand_dims(y_train, 1)
y_train = y_train.reshape((y_train.shape[0], 1))

print(x_train.shape)
print("Building features...")
x_train_nonans, removed_cols = bf.build_train_features(
    x=x_train, percentage=1, fill_nans=""
)
print(x_train_nonans.shape)

initial_w = f.initialize_weights(x_train_nonans, how="random")

print("Running gradient descent...")
w, loss = impl.mean_squared_error_gd(y_train, x_train_nonans, initial_w, parameters.iters, parameters.gamma)
# w, loss = impl.logistic_regression(y_train, x_train_nonans, initial_w, parameters.iters, parameters.gamma)

print(f"Loss: {loss} ")
