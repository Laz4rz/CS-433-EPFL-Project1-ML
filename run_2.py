import itertools
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
import dump.maxfunc as mf

from tqdm import tqdm
from src.model.Models import Models
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
        percentage=parameters.percentage,
        fill_nans=parameters.fill_nans,
        balance=parameters.balance,
        balance_scale=parameters.balance_scale,
        drop_calculated=parameters.drop_calculated,
        polynomial_expansion_degree=parameters.degree,
    )
    print(f"Size after build {x_train_nonans_balanced.shape}")

    x_train_full = bf.build_test_features(
        x=x_train,
        idx_calc_columns=idx_calc_columns,
        idx_nan_percent=idx_nan_percent,
        fill_nans=parameters.fill_nans,
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
x_train, x_test, y_train, train_ids, test_ids = hp.load_csv_data("data/")

y_train = np.expand_dims(y_train, 1)
y_train = y_train.reshape((y_train.shape[0], 1))


gammas = [0.1, 0.01, 0.001, 0.0001]
degrees = [1, 2]
how_inits = ["random", "zeros"]
fill_nans = ["mean", "random"]
percentages = [75, 99]
balance_scales = [1, 2]
drop_calculateds = [True, False]
balances = [True]

combinations = itertools.product(
    gammas,
    degrees,
    how_inits,
    fill_nans,
    percentages,
    balance_scales,
    drop_calculateds,
    balances,
)

results = {}

for combination in tqdm(combinations):
    parameters = Parameters(
        seed=42,
        lambda_=0.1,
        iters=750,
        gamma=combination[0],
        degree=combination[1],
        balance=combination[7],
        balance_scale=combination[5],
        drop_calculated=combination[6],
        percentage=combination[4],
        fill_nans=combination[3],
        how_init=combination[2],
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

    print("=" * 50)

    results[str(parameters)] = {
        "f1_training": f1_training,
        "acc_training": acc_training,
        "f1_full": f1_full,
        "acc_full": acc_full,
        "loss": loss,
    }


def get_best_f1_full_from_results(results: dict) -> float:
    best_f1_full = 0
    best_parameters = None
    for parameters, result in results.items():
        if result["f1_full"] > best_f1_full:
            best_f1_full = result["f1_full"]
            best_parameters = parameters
    return best_f1_full, best_parameters


# utilsf.create_submission(
#     x=x_test,
#     ids=test_ids,
#     w=w,
#     model=Models.LOGISTIC,
#     idx_calc_columns=idx_calc_columns,
#     idx_nan_percent=idx_nan_percent,
#     fill_nans="random",
#     filename="submission.csv",
# )

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
