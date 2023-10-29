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
import src.utils.functions as utilsf

from src.model.Models import Models
from src.utils.parameters import Parameters


parameters = Parameters(
    seed=42,
    lambda_=0.4,
    iters=1000,
    gamma=0.01,
    degree=1
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
    x=x_train, y=y_train, percentage=90, fill_nans="mean", balance=True, balance_scale=1, drop_calculated=True, polynomial_expansion_degree=parameters.degree
)
print(x_train_nonans_balanced.shape)

x_train_full = bf.build_test_features(x=x_train, idx_calc_columns=idx_calc_columns, idx_nan_percent=idx_nan_percent, fill_nans="mean", polynomial_expansion_degree=parameters.degree)

initial_w = f.initialize_weights(x_train_nonans_balanced, how="zeros")



print("Running cross validation...")
for gamma in [0.01]:
    print(f"Gamma = {gamma}")
    # acc, f1, w = train.run_cross_validation(x_train_nonans_balanced, y_train_balanced, 2, impl.mean_squared_error_gd, Models.LINEAR, max_iters=parameters.iters, gamma=gamma, initial_w=initial_w)
    # w, loss = impl.mean_squared_error_gd(y_train_balanced, x_train_nonans_balanced, initial_w, parameters.iters, gamma)
    w, loss = impl.reg_logistic_regression(y_train_balanced, x_train_nonans_balanced, 0.1, initial_w, parameters.iters, gamma)
    f1_full = ev.compute_f1_score(y_train, pred.compute_predictions_logistic(x_train_full, w))
    acc_full = ev.compute_accuracy(y_train, pred.compute_predictions_logistic(x_train_full, w))
    f1_balanced = ev.compute_f1_score(y_train_balanced, pred.compute_predictions_logistic(x_train_nonans_balanced, w))
    acc_balanced = ev.compute_accuracy(y_train_balanced, pred.compute_predictions_logistic(x_train_nonans_balanced, w))
    print(f"F1 score on full set: {f1_full}")
    print(f"Acc on training set: {acc_full}")
    print(f"F1 score on balanced set: {f1_balanced}")
    print(f"Acc on balanced set: {acc_balanced}")
    plt.hist(pred.compute_predictions_logistic(x_train_full, w), 100)
    plt.show()

utilsf.create_submission(
    x=x_test,
    ids=test_ids,
    w=w,
    model=Models.LOGISTIC,
    idx_calc_columns=idx_calc_columns,
    idx_nan_percent=idx_nan_percent,
    fill_nans="random",
    filename="submission.csv",
)

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
