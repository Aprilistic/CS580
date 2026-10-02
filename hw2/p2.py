"""CSC 480/580 Homework 2, Problem 2.

Part (a) uses a local CSV curated from the Wisconsin State Climatology Office
tables linked below. For each winter from 1855-56 through 2018-19, the CSV
keeps the row with a numeric Days of Ice Cover value and uses the first year
of the winter as x. It excludes duplicate rows with blank ice-day values.

Run with: uv run --with pandas --with matplotlib python p2.py
"""

# %% Problem 2: Linear regression

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a display.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


MENDOTA_URL = (
    "https://climatology.nelson.wisc.edu/first-order-station-climate-data/"
    "madison-climate/lake-ice/history-of-ice-freezing-and-thawing-on-lake-mendota/"
)
MONONA_URL = (
    "https://climatology.nelson.wisc.edu/first-order-station-climate-data/"
    "madison-climate/lake-ice/history-of-ice-freezing-and-thawing-on-lake-monona/"
)
DATA_FILE = Path(__file__).with_name("lake_ice_days_1855_2018.csv")


def curate_lake_data():
    """Return a DataFrame with aligned year and ice-day columns.

    Cover the years 1855-56 through 2018-19 and apply the assignment's rule
    for years with multiple freeze-thaw cycles.
    """
    data = pd.read_csv(
        DATA_FILE,
        dtype={"year": int, "mendota_days": int, "monona_days": int},
    )
    return data


def problem_2a(data):
    """Create both requested plots."""
    output_dir = Path(__file__).parent

    fig, ax = plt.subplots(figsize=(9, 4.5))
    data.plot(
        x="year", y=["mendota_days", "monona_days"], ax=ax, linewidth=1.4
    )
    ax.set(xlabel="Winter starting year", ylabel="Days of ice cover",
           title="Lake ice cover, 1855-56 to 2018-19")
    ax.grid(alpha=0.25)
    ax.legend(["Lake Mendota", "Lake Monona"])
    fig.tight_layout()
    lake_plot = output_dir / "p2a_lake_ice_days.png"
    fig.savefig(lake_plot, dpi=220)
    plt.close(fig)

    difference_data = data.assign(
        monona_minus_mendota=data["monona_days"] - data["mendota_days"]
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    difference_data.plot(
        x="year", y="monona_minus_mendota", ax=ax,
        color="tab:purple", linewidth=1.4, legend=False,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.9)
    ax.set(xlabel="Winter starting year",
           ylabel="Monona - Mendota (ice days)",
           title="Difference in lake ice cover, 1855-56 to 2018-19")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    difference_plot = output_dir / "p2a_lake_difference.png"
    fig.savefig(difference_plot, dpi=220)
    plt.close(fig)

    return lake_plot, difference_plot


def problem_2b(data):
    """Split by year and print training statistics for each lake."""
    split_year = 1970
    training_data = data.loc[data["year"] <= split_year].copy()
    test_data = data.loc[data["year"] > split_year].copy()

    for lake, column in (
        ("Mendota", "mendota_days"),
        ("Monona", "monona_days"),
    ):
        mean_days = training_data[column].mean()
        sample_sd = training_data[column].std(ddof=1)
        print(f"{lake}: mean={mean_days:.2f}, sample SD={sample_sd:.2f}")

    return training_data, test_data


def problem_2c(train_data):
    """Fit the closed-form OLS model with year and Monona ice days."""
    X = np.column_stack((
        np.ones(len(train_data)),
        train_data["year"].to_numpy(),
        train_data["monona_days"].to_numpy(),
    ))
    y = train_data["mendota_days"].to_numpy()

    beta = np.linalg.solve(X.T @ X, X.T @ y)
    return beta


def problem_2d(test_data, beta):
    """Calculate test-set mean squared error for the model in 2(c)."""
    X = np.column_stack((
        np.ones(len(test_data)),
        test_data["year"].to_numpy(),
        test_data["monona_days"].to_numpy(),
    ))
    y = test_data["mendota_days"].to_numpy()
    y_pred = X @ beta
    mse = np.mean((y - y_pred) ** 2)
    return mse


def problem_2e(data):
    """Fit the closed-form OLS model using year only."""
    X = np.column_stack((
        np.ones(len(data)),
        data["year"].to_numpy(),
    ))
    y = data["mendota_days"].to_numpy()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    return beta


if __name__ == "__main__":
    lake_data = curate_lake_data()
    lake_plot, difference_plot = problem_2a(lake_data)
    print(
        f"Loaded {len(lake_data)} aligned winters: "
        f"{lake_data['year'].iloc[0]}-{lake_data['year'].iloc[-1]}."
    )
    print(f"Saved: {lake_plot}")
    print(f"Saved: {difference_plot}")

    print("\nProblem 2(b): training data through 1970")
    train_data, test_data = problem_2b(lake_data)

    beta_2c = problem_2c(train_data)
    print(f"\nProblem 2(c): OLS coefficients for Mendota using Monona ice days")
    print(f"beta_0 (intercept): {beta_2c[0]:.4f}")
    print(f"beta_1 (year): {beta_2c[1]:.4f}")
    print(f"beta_2 (monona_days): {beta_2c[2]:.4f}")

    mse_2d = problem_2d(test_data, beta_2c)
    print(f"\nProblem 2(d): test-set MSE for the model in 2(c)")
    print(f"MSE: {mse_2d:.2f}")

    gamma_2e = problem_2e(train_data)
    print(f"\nProblem 2(e): OLS coefficients for Mendota using year only")
    print(f"gamma_0 (intercept): {gamma_2e[0]:.4f}")
    print(f"gamma_1 (year): {gamma_2e[1]:.4f}")


