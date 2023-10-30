import itertools
import numpy as np
import helpers as hp
import implementations as impl
import matplotlib.pyplot as plt
import src.utils.functions as f
import src.model.predict_model as pred
import src.evaluation.evaluation as ev
import src.evaluation.evaluation as ev
import src.features.build_features as bf

from tqdm import tqdm
from src.utils.parameters import Parameters


def split_train_test(x, y, ratio=0.8):
    """split the dataset based on the split ratio."""
    train_size = int(ratio * len(y))

    idx_0 = np.where(y == -1)[0]
    idx_1 = np.where(y == 1)[0]

    label_1_ratio = len(idx_1) / len(y)
    label_0_ratio = len(idx_0) / len(y)

    take_1 = int(train_size * label_1_ratio)
    take_0 = int(train_size * label_0_ratio)

    idx_train = np.random.choice(idx_1, take_1, replace=False)
    idx_train = np.append(idx_train, np.random.choice(idx_0, take_0, replace=False))

    idx_test = np.setdiff1d(np.arange(len(y)), idx_train)

    return x[idx_train], y[idx_train], x[idx_test], y[idx_test]


print("Loading data...")
x_train, x_test, y_train, train_ids, test_ids = hp.load_csv_data("data/")

y_train = np.expand_dims(y_train, 1)
y_train = y_train.reshape((y_train.shape[0], 1))

gammas = [0.15]
degrees = [5]
how_inits = ["zeros"]
fill_nans = ["mean"]
percentages_col = [90]
percentages_row = [50]
balance_scales = [3]
drop_calculateds = [True]
balances = [True]
nums = [0]

combinations = itertools.product(
    gammas,
    degrees,
    how_inits,
    fill_nans,
    percentages_col,
    balance_scales,
    drop_calculateds,
    balances,
    nums,
    percentages_row,
)

results = {}

for combination in tqdm(list(combinations)):
    parameters = Parameters(
        seed=42,
        lambda_=0.1,
        iters=600,
        gamma=combination[0],
        degree=combination[1],
        how_init=combination[2],
        fill_nans=combination[3],
        percentage_col=combination[4],
        percentage_row=combination[9],
        balance_scale=combination[5],
        drop_calculated=combination[6],
        balance=combination[7],
        num=combination[8],
        drop_outliers=9,
    )

    f.set_random_seed(parameters.seed)

    x_train_balanced, y_train_balanced, _, _, x_train_full, initial_w = bf.build_all(
        x_train=x_train, y_train=y_train, parameters=parameters
    )

    xtr, ytr, xte, yte = split_train_test(x_train_balanced, y_train_balanced, ratio=0.8)

    print(f"GD for {parameters}")
    w, loss = impl.logistic_regression(
        ytr,
        xtr,
        initial_w,
        parameters.iters,
        parameters.gamma,
    )

    results = ev.full_evaluation(
        x_train=x_train_balanced,
        y_train=y_train_balanced,
        x_train_full=x_train_full,
        y_train_full=y_train,
        x_test=xte,
        y_test=yte,
        w=w,
        results=results,
        parameters=parameters,
        compute_predictions_func=pred.compute_predictions_logistic,
        loss=loss,
    )

f.pickle_results(results, "results/results_logistic_regression_for_boxplots.pickle")
