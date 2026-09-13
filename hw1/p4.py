### p4.py

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Constants
# ============================================================

A = np.array([
    [0.1, 0.2, 0.2],
    [0.2, 0.4, 0.8],
    [0.2, 0.8, 0.9],
], dtype=float)

M_VALUES = [10, 30, 100, 300, 1000, 3000]
K_VALUES = [1, 2, 4, 8, 16, 32, 64]


# ============================================================
# Distribution D
# ============================================================

def cell_indices(X):
    """
    Convert points X in [0, 1]^2 into the corresponding
    3x3 cell indices of matrix A.

    Returns indices 0, 1, or 2.
    """
    X = np.asarray(X, dtype=float)

    i = np.minimum((3 * X[:, 0]).astype(int), 2)
    j = np.minimum((3 * X[:, 1]).astype(int), 2)

    return i, j


def draw_samples(m, rng):
    """
    Draw m i.i.d. samples from D.

    X1, X2 ~ Uniform[0, 1]
    Y | X=x ~ Bernoulli(A[i(x), j(x)])
    """
    X = rng.uniform(
        0,
        1,
        size=(m, 2)
    )

    i, j = cell_indices(X)

    p_y1 = A[i, j]

    y = (
        rng.uniform(0, 1, size=m) < p_y1
    ).astype(int)

    return X, y


# ============================================================
# Bayes classifier
# ============================================================

def bayes_predict(X):
    """
    Predict 1 if P(Y=1 | X=x) >= 0.5.
    Otherwise predict 0.
    """
    i, j = cell_indices(X)

    return (
        A[i, j] >= 0.5
    ).astype(int)


def bayes_error_rate():
    """
    Each cell has probability 1/9.

    In a cell with probability p,
    the Bayes error is min(p, 1-p).
    """
    return np.mean(
        np.minimum(
            A,
            1 - A
        )
    )


# ============================================================
# k-NN
# ============================================================

def predict_for_many_k(
    X_train,
    y_train,
    X_test,
    k_values
):
    """
    Predict labels for multiple k values while computing
    nearest neighbors only once.

    If k > number of training samples,
    use k = number of training samples.
    """

    X_train = np.asarray(
        X_train,
        dtype=float
    )

    y_train = np.asarray(
        y_train,
        dtype=int
    )

    X_test = np.asarray(
        X_test,
        dtype=float
    )

    n_train = len(X_train)

    effective_k = {
        k: min(int(k), n_train)
        for k in k_values
    }

    max_k = max(
        effective_k.values()
    )

    # --------------------------------------------------------
    # Squared Euclidean distances
    #
    # ||x - z||^2
    # = ||x||^2 + ||z||^2 - 2 x^T z
    # --------------------------------------------------------

    test_norm = np.sum(
        X_test ** 2,
        axis=1,
        keepdims=True
    )

    train_norm = np.sum(
        X_train ** 2,
        axis=1
    )

    dist2 = (
        test_norm
        + train_norm[None, :]
        - 2 * X_test @ X_train.T
    )

    # Protect against tiny negative values
    # caused by floating point error.
    np.maximum(
        dist2,
        0,
        out=dist2
    )

    # --------------------------------------------------------
    # Find max_k nearest neighbors
    # --------------------------------------------------------

    candidate_idx = np.argpartition(
        dist2,
        kth=max_k - 1,
        axis=1
    )[:, :max_k]

    candidate_dist = np.take_along_axis(
        dist2,
        candidate_idx,
        axis=1
    )

    # Sort candidates from nearest to farthest.
    order = np.argsort(
        candidate_dist,
        axis=1
    )

    nearest_idx = np.take_along_axis(
        candidate_idx,
        order,
        axis=1
    )

    nearest_labels = y_train[
        nearest_idx
    ]

    # Number of class-1 labels among
    # first 1, 2, ..., max_k neighbors.
    cumulative_ones = np.cumsum(
        nearest_labels,
        axis=1
    )

    predictions = {}

    for k in k_values:

        k_eff = effective_k[k]

        votes_for_1 = cumulative_ones[
            :, k_eff - 1
        ]

        # Majority vote.
        # Tie goes to class 0.
        predictions[k] = (
            votes_for_1 > k_eff / 2
        ).astype(int)

    return predictions


# ============================================================
# Simple k-NN classifier
# ============================================================

class KNNClassifier:

    def __init__(self, k=1):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):

        self.X_train = np.asarray(
            X,
            dtype=float
        )

        self.y_train = np.asarray(
            y,
            dtype=int
        )

        return self

    def predict(self, X):

        if self.X_train is None:
            raise ValueError(
                "Call fit() before predict()."
            )

        predictions = predict_for_many_k(
            self.X_train,
            self.y_train,
            X,
            [self.k]
        )

        return predictions[self.k]


# ============================================================
# Parts (b) and (c)
# ============================================================

def run_experiments(
    X_test,
    y_test,
    rng,
    n_trials=5
):
    """
    Run n_trials independent training experiments.

    The same fixed test set is reused for every trial.

    Returns:
        avg_errors:
            mean test error across trials

        std_errors:
            standard deviation of test error across trials
    """

    errors = np.zeros(
        (
            n_trials,
            len(K_VALUES),
            len(M_VALUES)
        ),
        dtype=float
    )

    for trial in range(n_trials):

        print(
            f"Trial {trial + 1}/{n_trials}"
        )

        # Draw one training set of size 3000.
        X_S, y_S = draw_samples(
            3000,
            rng
        )

        for m_idx, m in enumerate(M_VALUES):

            # Use first m samples.
            X_train = X_S[:m]
            y_train = y_S[:m]

            predictions = predict_for_many_k(
                X_train,
                y_train,
                X_test,
                K_VALUES
            )

            for k_idx, k in enumerate(K_VALUES):

                y_pred = predictions[k]

                errors[
                    trial,
                    k_idx,
                    m_idx
                ] = np.mean(
                    y_pred != y_test
                )

    # Mean across trials
    avg_errors = np.mean(
        errors,
        axis=0
    )

    # Standard deviation across trials
    std_errors = np.std(
        errors,
        axis=0,
        ddof=1
    )

    return avg_errors, std_errors


# ============================================================
# Part (b)
# 4-NN learning curve with standard deviation
# ============================================================

def plot_part_b(
    avg_errors,
    std_errors,
    bayes_error
):
    k_idx = K_VALUES.index(4)

    plt.figure(
        figsize=(7, 5)
    )

    plt.errorbar(
        M_VALUES,
        avg_errors[k_idx],
        yerr=std_errors[k_idx],
        marker="o",
        capsize=4,
        label="4-NN average test error ± std"
    )

    plt.axhline(
        bayes_error,
        linestyle="--",
        label=(
            f"Bayes error = "
            f"{bayes_error:.3f}"
        )
    )

    plt.xscale("log")

    plt.xlabel(
        "Training set size m"
    )

    plt.ylabel(
        "Test error rate"
    )

    plt.title(
        "Part (b): Learning Curve for 4-NN"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "part_b_knn_learning_curve.png",
        dpi=200
    )

    plt.show()


# ============================================================
# Part (c)
# k-NN curves for different k
# ============================================================

def plot_part_c(
    avg_errors,
    bayes_error
):
    plt.figure(
        figsize=(8, 6)
    )

    for k_idx, k in enumerate(K_VALUES):

        plt.plot(
            M_VALUES,
            avg_errors[k_idx],
            marker="o",
            label=f"k={k}"
        )

    plt.axhline(
        bayes_error,
        linestyle="--",
        label=(
            f"Bayes error = "
            f"{bayes_error:.3f}"
        )
    )

    plt.xscale("log")

    plt.xlabel(
        "Training set size m"
    )

    plt.ylabel(
        "Test error rate"
    )

    plt.title(
        "Part (c): k-NN Learning Curves"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "part_c_knn_learning_curves.png",
        dpi=200
    )

    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    rng = np.random.default_rng(42)

    # --------------------------------------------------------
    # Part (a)
    # --------------------------------------------------------

    bayes_error = bayes_error_rate()

    print("Bayes classifier:")

    print(
        "  Predict 1 in cells "
        "(2,3), (3,2), and (3,3)."
    )

    print(
        "  Predict 0 in all other cells."
    )

    print(
        f"Bayes error rate = "
        f"{bayes_error:.3f}"
    )

    print()

    # --------------------------------------------------------
    # Fixed test set
    # --------------------------------------------------------

    X_test, y_test = draw_samples(
        10_000,
        rng
    )

    # --------------------------------------------------------
    # Parts (b) and (c)
    # --------------------------------------------------------

    avg_errors, std_errors = run_experiments(
        X_test,
        y_test,
        rng,
        n_trials=5
    )

    # --------------------------------------------------------
    # Print average errors and standard deviations
    # --------------------------------------------------------

    print(
        "\nAverage test errors "
        "(mean ± std):"
    )

    header = "m".ljust(8)

    for k in K_VALUES:
        header += f"k={k:<16}"

    print(header)

    for m_idx, m in enumerate(M_VALUES):

        row = f"{m:<8}"

        for k_idx in range(len(K_VALUES)):

            mean = avg_errors[
                k_idx,
                m_idx
            ]

            std = std_errors[
                k_idx,
                m_idx
            ]

            row += (
                f"{mean:.4f} ± "
                f"{std:.4f}".ljust(10)
            )

        print(row)

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_part_b(
        avg_errors,
        std_errors,
        bayes_error
    )

    plot_part_c(
        avg_errors,
        bayes_error
    )