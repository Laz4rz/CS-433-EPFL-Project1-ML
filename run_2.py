import numpy as np
import helpers as hp
import matplotlib.pyplot as plt
import implementations as impl
import src.utils.constants as c
import src.utils.functions as f
import src.model.predict_model as pm
import src.evaluation.evaluation as ev
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

y_train = np.expand_dims(y_train, 1)
y_train = y_train.reshape((y_train.shape[0], 1))

print("Building features...")
x_train_nonans, removed_cols = bf.build_train_features(
    x=x_train, percentage=1, fill_nans=""
)
lambda_ = 0.1
max_iters = 50
threshold = 1e-8
gammas = np.linspace(0.1, 0.5, 5)
initial_w = np.zeros((x_train_nonans.shape[1], 1))

print("Running gradient descent...")

best_loss = float("inf")
best_gamma = 0
best_w = initial_w
computes_loss = []
computes_w = []
for gamma in gammas:
    w_mean_squared_error_gd, loss_mean_squared_error_gd = impl.reg_logistic_regression(
        y_train, x_train_nonans, lambda_, initial_w, max_iters, gamma
    )
    computes_loss.append(loss_mean_squared_error_gd)
    computes_w.append(w_mean_squared_error_gd)
    if loss_mean_squared_error_gd < best_loss:
        best_loss = loss_mean_squared_error_gd
        best_w = w_mean_squared_error_gd
        best_gamma = gamma

print("Best gamma: {gamma}".format(gamma=best_gamma))
print("Best loss: {loss}".format(loss=best_loss))
print("Best w: {w}".format(w=best_w))

pred = pm.compute_predictions_linear(x_train_nonans, w_mean_squared_error_gd)
f1 = ev.compute_f1_score(y_train, pred)

# print("Loss:{loss} ".format(loss=loss_mean_squared_error_gd))
print("F1:{f1} ".format(f1=f1))
