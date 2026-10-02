"""CSC 480/580 Homework 2, Problem 3: k-NN ROC curves.

The distribution and Euclidean-distance calculation come from the Homework 1
Problem 4 implementation in ../hw1/p4.py.

Run with: uv run --with numpy --with matplotlib python p3.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Conditional probabilities P(Y=1 | X in cell) from Homework 1.
A = np.array([
    [0.1, 0.2, 0.2],
    [0.2, 0.4, 0.8],
    [0.2, 0.8, 0.9],
])


def sample_distribution(n_samples, rng=None):
    """Draw samples from the Homework 1 distribution D."""
    if rng is None:
        rng = np.random.default_rng()
    X = rng.uniform(0, 1, size=(n_samples, 2))
    cells = np.minimum((3 * X).astype(int), 2)
    probabilities = A[cells[:, 0], cells[:, 1]]
    y = (rng.uniform(0, 1, size=n_samples) < probabilities).astype(int)
    return X, y


class KNNClassifier:
    """Homework 1 k-NN, extended to return scores and use a threshold."""

    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store the training features and labels."""
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=int)
        if not 1 <= self.k <= len(self.X_train):
            raise ValueError("k must be between 1 and the training set size")
        return self

    # 3.1.(a)
    def predict_proba(self, X):
        """Return the fraction of positive labels among the k nearest neighbors."""
        if self.X_train is None:
            raise ValueError("Call fit() before predict_proba().")
        X = np.asarray(X, dtype=float)
        # Same squared Euclidean-distance identity used in Homework 1.
        distances_sq = (
            np.sum(X**2, axis=1, keepdims=True)
            + np.sum(self.X_train**2, axis=1)[None, :]
            - 2 * X @ self.X_train.T
        )
        np.maximum(distances_sq, 0, out=distances_sq)
        neighbors = np.argpartition(
            distances_sq, kth=self.k - 1, axis=1
        )[:, :self.k]
        return self.y_train[neighbors].mean(axis=1)

    # 3.1.(b)
    def predict(self, X, threshold=0.5):
        """Predict 1 when the positive-neighbor fraction reaches threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)


# 3.2
def compute_rates(y_true, y_scores, threshold):
    """Compute (FPR, TPR) from the four confusion-matrix counts."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    predicted_positive = y_scores >= threshold
    actual_positive = y_true == 1
    actual_negative = y_true == 0

    tp = np.count_nonzero(actual_positive & predicted_positive)
    fn = np.count_nonzero(actual_positive & ~predicted_positive)
    fp = np.count_nonzero(actual_negative & predicted_positive)
    tn = np.count_nonzero(actual_negative & ~predicted_positive)
    if tp + fn == 0 or fp + tn == 0:
        raise ValueError("ROC rates require both classes in the test set")
    return fp / (fp + tn), tp / (tp + fn)


def compute_roc_curve(y_true, y_scores):
    """Sweep distinct scores from high to low; return (FPR, TPR, thresholds)."""
    y_scores = np.asarray(y_scores)
    # An initial threshold above every score supplies the (0, 0) endpoint.
    thresholds = np.r_[np.inf, np.unique(y_scores)[::-1]]
    fpr = np.empty(len(thresholds))
    tpr = np.empty(len(thresholds))
    for i, threshold in enumerate(thresholds):
        fpr[i], tpr[i] = compute_rates(y_true, y_scores, threshold)
    return fpr, tpr, thresholds


def compute_auc(fpr, tpr):
    """Calculate trapezoidal area beneath an ROC curve."""
    return np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2)


def problem_3():
    """Draw the requested samples and save all four ROC curves in one figure."""
    rng = np.random.default_rng(42)
    X_train, y_train = sample_distribution(3_000, rng)
    X_test, y_test = sample_distribution(500, rng)

    fig, ax = plt.subplots(figsize=(7, 6))
    aucs = {}
    for k in (1, 5, 25, 101):
        model = KNNClassifier(k).fit(X_train, y_train)
        scores = model.predict_proba(X_test)
        fpr, tpr, _ = compute_roc_curve(y_test, scores)
        aucs[k] = compute_auc(fpr, tpr)
        ax.step(fpr, tpr, where="post", label=f"k={k}, AUC={aucs[k]:.3f}")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guessing")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate",
           ylabel="True positive rate", title="k-NN ROC curves")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path = Path(__file__).with_name("p3_roc_curves.png")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return aucs, output_path


if __name__ == "__main__":
    aucs, plot_path = problem_3()
    for k, auc in aucs.items():
        print(f"k={k}: AUC={auc:.4f}")
    print(f"Saved: {plot_path}")
