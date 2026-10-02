"""CSC 480/580 Homework 2, Problem 4: ridge regression (CSC 580 only).

Run with: uv run --with numpy --with pandas --with matplotlib python p4.py
"""

# %% Problem 4: Ridge regression (CSC 580 students only)

from p2 import curate_lake_data, problem_2b

import numpy as np


def problem_4b(years, mendota_days, monona_days):
    """Fit ridge regression on the Problem 2(c) training data with lambda = 1."""
    n = len(years)
    X = np.column_stack((np.ones(n), years, monona_days))
    y = mendota_days

    A = np.diag([0, 1, 1])  # Leave the intercept unpenalized.
    lambda_value = 1.0

    beta = np.linalg.solve(
        X.T @ X + n * lambda_value * A,
        X.T @ y,
    )
    return beta


def problem_4c(years, mendota_days, monona_days):
    """Return mean validation MSEs and the best lambda from five-fold CV."""
    n = len(years)
    X = np.column_stack((np.ones(n), years, monona_days))
    y = np.asarray(mendota_days)
    A = np.diag([0, 1, 1])
    lambda_values = (0.001, 0.01, 0.1, 1.0, 10.0)

    # Shuffle once so every lambda uses the same reproducible folds.
    rng = np.random.default_rng(42)
    folds = np.array_split(rng.permutation(n), 5)

    mean_validation_errors = {}
    for lambda_value in lambda_values:
        fold_errors = []
        for fold_index, validation_indices in enumerate(folds):
            training_indices = np.concatenate([
                fold for i, fold in enumerate(folds) if i != fold_index
            ])
            X_train, y_train = X[training_indices], y[training_indices]
            X_validation = X[validation_indices]
            y_validation = y[validation_indices]

            # n in the ridge formula is this fold's training sample count.
            n_train = len(training_indices)
            beta = np.linalg.solve(
                X_train.T @ X_train + n_train * lambda_value * A,
                X_train.T @ y_train,
            )
            predictions = X_validation @ beta
            fold_errors.append(np.mean((predictions - y_validation) ** 2))

        mean_validation_errors[lambda_value] = np.mean(fold_errors)

    best_lambda = min(mean_validation_errors, key=mean_validation_errors.get)
    return mean_validation_errors, best_lambda


if __name__ == "__main__":
    lake_data = curate_lake_data()
    train_data, test_data = problem_2b(lake_data)

    # Use the same training observations as Problem 2(c).
    years = train_data["year"].to_numpy()
    mendota_days = train_data["mendota_days"].to_numpy()
    monona_days = train_data["monona_days"].to_numpy()

    print(f"Loaded {len(train_data)} training winters for Problem 4.")

    beta_4b = problem_4b(years, mendota_days, monona_days)
    print(f"\nProblem 4(b): Ridge coefficients (lambda=1.0)")
    print(f"beta_0 (intercept): {beta_4b[0]:.4f}")
    print(f"beta_1 (year): {beta_4b[1]:.4f}")
    print(f"beta_2 (monona_days): {beta_4b[2]:.4f}")

    cv_errors, best_lambda = problem_4c(years, mendota_days, monona_days)
    print("\nProblem 4(c): five-fold cross-validation (seed=42)")
    for lambda_value, mean_mse in cv_errors.items():
        print(f"lambda={lambda_value:g}: mean validation MSE={mean_mse:.4f}")
    print(f"Selected lambda: {best_lambda:g}")
