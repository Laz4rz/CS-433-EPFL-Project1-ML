import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import helpers as hp
import implementations as impl

print("Loading data...")
x_train, x_test, y_train, train_ids, test_ids = hp.load_csv_data("data/")

y_train = np.expand_dims(y_train, 1)
y_train = y_train.reshape((y_train.shape[0],1))

import src.utils.constants as c
import src.features.build_features as bf

x_train_nonans, removed_cols = bf.build_train_features(data=x_train, percentage=0.2)

y0_idx = np.random.choice(np.arange(0, len(y_train))[(y_train == 0).squeeze()], np.sum(y_train[y_train == 1]))
balanced_idx = np.concatenate([y0_idx, np.arange(0, len(y_train))[(y_train == 1).squeeze()]])

balanced_idx = np.sort(balanced_idx)

y_balanced = y_train[balanced_idx]
x_balanced = x_train_nonans[balanced_idx]

lambda_ = 0.1                                      # regularization parameter
max_iters = 50                                     # max number of iterations 
threshold = 1e-8                                   # threshold for stopping criterion
initial_w = np.ones((x_train_nonans.shape[1], 1)) # initial weights

gammas = np.array([10**(-i) for i in range(2, 6)])
iters = [1, 2, 5, 10, 25, 50]
losses = []
weights = []

print("Running GD...")
for gamma in gammas:
    for iterations in iters:
        last_w, loss_mean_squared_error_gd = impl.mean_squared_error_gd(y_balanced, x_balanced, initial_w, iterations, gamma)
        losses.append(loss_mean_squared_error_gd)
        weights.append(last_w)

losses = pd.DataFrame(index=gammas, columns=iters, data=np.array(losses).reshape(len(gammas), len(iters)))
weights = pd.DataFrame(index=gammas, columns=iters, data=np.array(weights).reshape(len(gammas), len(iters)))
sns.heatmap(pd.DataFrame(index=gammas, columns=iters, data=np.array(losses).reshape(len(gammas), len(iters))), annot=True)

