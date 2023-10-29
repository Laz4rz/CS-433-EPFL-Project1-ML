![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
# CS-433 Machine Learning - Project 1 Report

Group:
- Lorenzo Drudi - 367980
- Mikolaj Boronski -
- Olena Zavertiaieva - 364500

> The `run.py` file contains the code for the best model we obtained.

#### Implemented Algorithms:
- `Linear regression using gradient descent`
- `Linear regression using stochastic gradient descent`
- `Least squares regression using normal equation`
- `Ridge regression using normal equation`
- `Logistic regression using gradient descent`
- `Regularized logistic regression using gradient descent`

All the algorithms are inside the `implementations.py` file.

#### Project structure

- `implementations.py`: implementations of all the ML algorithms.
- `src/features`: functions for data cleaning and feature processing (NaN replacing, downsampling, columns removal, outliers removal).
- `src/model`: functions for training and getting the predictions (k-fold cross-validation).
- `src/evaluation`: evaluation metrics (rsme, accuracy, f1).
- `src/utils`: utility functions (batch iteration, submission creation, weights generation).

#### Useful functions implemented
- `Data Cleaning and Feature Processing`:
  - Drop calculated features.
  - Drop rows with more than NaN% of values.
  - Remove outliers.
  - Replace NaN with mean, most frequent, or random uniform distribution between min and max.
  - Downsampling (labels balancing).
  - Standardization (z-score).
  - Polynomial expansion.
- `Training process`:
  - K-fold cross-validation.
- `Evaluation metrics`:
  - F1 score.
  - Accuracy.
  - RSME.  
