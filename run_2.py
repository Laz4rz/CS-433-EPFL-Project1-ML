import numpy as np
import helpers as hp
import matplotlib.pyplot as plt
import implementations as impl
import src.evaluation.evaluation as ev
import src.utils.constants as c
import src.utils.functions as f
import src.model.predict_model as pm
import src.evaluation.evaluation as ev
import src.features.build_features as bf
import src.model.predict_model as pred
import src.model.train_model as train

from src.model.Models import Models
from src.utils.parameters import Parameters


parameters = Parameters(
    seed=42,
    lambda_=0.1,
    iters=1000,
    gamma=0.001,
)

f.set_random_seed(parameters.seed)

print(parameters)

print("Loading data...")
x_train, x_test, y_train, train_ids, test_ids = hp.load_csv_data("data/")

y_train = np.expand_dims(y_train, 1)
y_train = y_train.reshape((y_train.shape[0], 1))

print(x_train.shape)
print("Building features...")
x_train_nonans_balanced, y_train_balanced, idx_calc_columns, idx_nan_percent = bf.build_train_features(
    x=x_train, y=y_train, percentage=90, fill_nans="most_freq", balance=True, balance_scale=1
)
print(x_train_nonans_balanced.shape)

initial_w = f.initialize_weights(x_train_nonans_balanced, how="random")
x_train_full = bf.build_test_features(x=x_train, idx_calc_columns=idx_calc_columns, idx_nan_percent=idx_nan_percent, fill_nans="random")

for gamma in [10e-6, 10e-5, 10e-4]:
    print(f"Gamma = {gamma}")
    acc, f1, w = train.run_cross_validation(x_train_nonans_balanced, y_train_balanced, 2, impl.mean_squared_error_gd, Models.LINEAR, max_iters=parameters.iters, gamma=gamma, initial_w=initial_w)
    # w, loss = impl.mean_squared_error_gd(y_train_balanced, x_train_nonans_balanced, initial_w, parameters.iters, gamma)
    f1_training = ev.compute_f1_score(y_train_balanced, pred.compute_predictions_linear(x_train_nonans_balanced, w))
    print(f"F1 score on full set: {f1_training}")
    print(f"Acc on training set: {acc}")
    plt.hist(pred.compute_predictions_linear(x_train_full, w), 100)
    plt.show()



# print("Running gradient descent...")
# w, loss = impl.mean_squared_error_gd(y_train_balanced, x_train_nonans_balanced, initial_w, parameters.iters, parameters.gamma)
# # w, loss = impl.logistic_regression(y_train_balanced, x_train_nonans_balanced, initial_w, parameters.iters, parameters.gamma)

# print("Balanced training set:")
# f1_training = ev.compute_f1_score(y_train_balanced, pred.compute_predictions_linear(x_train_nonans_balanced, w))
# print(f"F1 score on training set: {f1_training}")
# print(f"Loss on training set: {loss}")

# print("\n")

# print("Full training set:")
# x_train_full = bf.build_test_features(x=x_train, idx_calc_columns=idx_calc_columns, idx_nan_percent=idx_nan_percent, fill_nans="random")
# f1_training_full = ev.compute_f1_score(y_train, pred.compute_predictions_linear(x_train_full, w))
# print(f"F1 score on training set: {f1_training_full}")
