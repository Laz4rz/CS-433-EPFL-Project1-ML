import numpy as np
import helpers as hp
import implementations as impl
import matplotlib.pyplot as plt
import src.utils.functions as f
import src.utils.constants as c
import src.evaluation.evaluation as ev
import src.model.predict_model as pred
import src.features.build_features as bf
from src.utils.parameters import Parameters


def build_all(x_train: np.ndarray, y_train: np.ndarray, parameters: Parameters):
    print(f"Building features for {parameters}")
    print(f"Size before build {x_train.shape}")
    (
        x_train_nonans_balanced,
        y_train_balanced,
        idx_calc_columns,
        idx_nan_percent,
    ) = bf.build_train_features(
        x=x_train,
        y=y_train,
        percentage_col=parameters.percentage_col,
        percentage_row=parameters.percentage_row,
        fill_nans=parameters.fill_nans,
        num=parameters.num,
        balance=parameters.balance,
        balance_scale=parameters.balance_scale,
        drop_calculated=parameters.drop_calculated,
        polynomial_expansion_degree=parameters.degree,
        drop_outliers=parameters.drop_outliers,
    )
    print(f"Size after build {x_train_nonans_balanced.shape}")

    x_train_full = bf.build_test_features(
        x=x_train,
        idx_calc_columns=idx_calc_columns,
        idx_nan_percent=idx_nan_percent,
        fill_nans=parameters.fill_nans,
        num=parameters.num,
        polynomial_expansion_degree=parameters.degree,
    )

    initial_w = f.initialize_weights(x_train_nonans_balanced, how=parameters.how_init)

    return (
        x_train_nonans_balanced,
        y_train_balanced,
        idx_calc_columns,
        idx_nan_percent,
        x_train_full,
        initial_w,
    )


print("Loading data...")
x_train, x_test, y_train, train_ids, test_ids = hp.load_csv_data(c.DATA_PATH)
y_train = np.expand_dims(y_train, 1)
y_train = y_train.reshape((y_train.shape[0], 1))

parameters = Parameters(
    seed=42,
    lambda_=6e-3,
    iters=5000,
    gamma=0.15,
    batch_size=32,
    degree=1,
    balance=False,
    balance_scale=1,
    drop_calculated=False,
    percentage_col=100,
    percentage_row=100,
    fill_nans="with_num",
    num=0,
    how_init="zeros",
    drop_outliers=None,
)

f.set_random_seed(parameters.seed)

x_train_balanced, y_train_balanced, _, _, x_train_full, initial_w = build_all(
    x_train=x_train, y_train=y_train, parameters=parameters
)
print(f"log reg for {parameters}")

w, loss = impl.logistic_regression(
    y_train_balanced,
    x_train_balanced,
    initial_w,
    parameters.iters,
    parameters.gamma,
)

print("\nBalanced training set:")

f1_training = ev.compute_f1_score(
    y_train_balanced, pred.compute_predictions_logistic(x_train_balanced, w)
)
acc_training = ev.compute_accuracy(
    y_train_balanced, pred.compute_predictions_logistic(x_train_balanced, w)
)
print(f"F1 score on training set: {f1_training}")
print(f"Accuracy on training set: {acc_training}")

print("\nFull training set:")
f1_full = ev.compute_f1_score(
    y_train, pred.compute_predictions_logistic(x_train_full, w)
)
acc_full = ev.compute_accuracy(
    y_train, pred.compute_predictions_logistic(x_train_full, w)
)
print(f"F1 score on training set: {f1_full}")
print(f"Accuracy on training set: {acc_full}")

print(f"\n Loss on training set: {loss}")
