### p5.py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

import p4


# ============================================================
# Constants
# ============================================================

M_VALUES = p4.M_VALUES
K_VALUES = p4.K_VALUES

DEPTH_VALUES = [1, 2, 3, 4, 5, 6]

BAYES_ERROR = p4.bayes_error_rate()


# ============================================================
# General helpers
# ============================================================

def error_rate(y_true, y_pred):
    """
    Classification error rate.
    """
    return np.mean(y_true != y_pred)


def valid_k_values(n_train):
    """
    If k is larger than the available training set,
    force k to equal the number of training examples.

    Duplicate resulting k values are removed.

    Example:
        n_train = 10
        [1, 2, 4, 8, 16, 32, 64]
        ->
        [1, 2, 4, 8, 10]
    """
    return sorted({
        min(k, n_train)
        for k in K_VALUES
    })


def choose_best(errors):
    """
    Select hyperparameter with smallest error.

    If multiple choices have the same error,
    choose the smaller hyperparameter.
    """
    return min(
        errors,
        key=lambda x: (errors[x], x)
    )


# ============================================================
# Problem 5(a)
#
# Hyperparameter tuning for k-NN
# ============================================================

def tune_k_training_error(
    X_train,
    y_train
):
    """
    Tune k using training-set error.
    """

    k_values = valid_k_values(
        len(X_train)
    )

    predictions = p4.predict_for_many_k(
        X_train,
        y_train,
        X_train,
        k_values
    )

    errors = {
        k: error_rate(
            y_train,
            predictions[k]
        )
        for k in k_values
    }

    return choose_best(errors)


def tune_k_holdout(
    X_train,
    y_train,
    seed=0
):
    """
    Tune k using a 20% holdout validation set.
    """

    rng = np.random.default_rng(seed)

    n = len(X_train)

    indices = rng.permutation(n)

    split = int(0.8 * n)

    train_idx = indices[:split]
    val_idx = indices[split:]

    X_subtrain = X_train[train_idx]
    y_subtrain = y_train[train_idx]

    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    k_values = valid_k_values(
        len(X_subtrain)
    )

    predictions = p4.predict_for_many_k(
        X_subtrain,
        y_subtrain,
        X_val,
        k_values
    )

    errors = {
        k: error_rate(
            y_val,
            predictions[k]
        )
        for k in k_values
    }

    return choose_best(errors)


def tune_k_cv(
    X_train,
    y_train,
    num_folds=5,
    seed=0
):
    """
    Tune k using 5-fold cross validation.
    """

    rng = np.random.default_rng(seed)

    n = len(X_train)

    indices = rng.permutation(n)

    folds = np.array_split(
        indices,
        num_folds
    )

    # Smallest number of training examples
    # available in any fold.
    min_fold_train_size = min(
        n - len(fold)
        for fold in folds
    )

    k_values = valid_k_values(
        min_fold_train_size
    )

    errors = {
        k: []
        for k in k_values
    }

    for fold_idx in range(num_folds):

        val_idx = folds[fold_idx]

        train_idx = np.concatenate([
            folds[j]
            for j in range(num_folds)
            if j != fold_idx
        ])

        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]

        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        predictions = p4.predict_for_many_k(
            X_fold_train,
            y_fold_train,
            X_val,
            k_values
        )

        for k in k_values:

            fold_error = error_rate(
                y_val,
                predictions[k]
            )

            errors[k].append(
                fold_error
            )

    mean_errors = {
        k: np.mean(errors[k])
        for k in k_values
    }

    return choose_best(
        mean_errors
    )


def tune_k_test_error(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Tune k directly using the test set.

    This is not valid in a real ML setting.
    It is used here only as the oracle method.
    """

    k_values = valid_k_values(
        len(X_train)
    )

    predictions = p4.predict_for_many_k(
        X_train,
        y_train,
        X_test,
        k_values
    )

    errors = {
        k: error_rate(
            y_test,
            predictions[k]
        )
        for k in k_values
    }

    best_k = choose_best(
        errors
    )

    return best_k, errors


def problem_5a(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Run Problem 5(a).

    One training set of size 3000 is used.
    For each m, use the first m examples.
    """

    results = []

    print()
    print("=" * 90)
    print("Problem 5(a): Hyperparameter Tuning")
    print("=" * 90)

    for m in M_VALUES:

        X_m = X_train[:m]
        y_m = y_train[:m]

        # ----------------------------------------------------
        # Tune k
        # ----------------------------------------------------

        k_training = tune_k_training_error(
            X_m,
            y_m
        )

        k_holdout = tune_k_holdout(
            X_m,
            y_m
        )

        k_cv = tune_k_cv(
            X_m,
            y_m
        )

        # Test-set tuning also gives all test errors,
        # so we only calculate test predictions once.
        k_oracle, test_errors = tune_k_test_error(
            X_m,
            y_m,
            X_test,
            y_test
        )

        oracle_error = test_errors[
            k_oracle
        ]

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        methods = {
            "Training error": k_training,
            "20% holdout": k_holdout,
            "5-fold CV": k_cv,
            "Test oracle": k_oracle
        }

        print()
        print(f"m = {m}")
        print("-" * 90)

        print(
            f"{'Method':<20}"
            f"{'Tuned k':<12}"
            f"{'Test error':<15}"
            f"{'|k - oracle|':<16}"
            f"{'Error gap':<12}"
        )

        print("-" * 90)

        result = {
            "m": m,
            "oracle_k": k_oracle,
            "oracle_error": oracle_error
        }

        for method, k in methods.items():

            # k chosen by holdout / CV can always be
            # applied to the full m-point dataset.
            if k not in test_errors:

                prediction = p4.predict_for_many_k(
                    X_m,
                    y_m,
                    X_test,
                    [k]
                )[k]

                test_error = error_rate(
                    y_test,
                    prediction
                )

            else:
                test_error = test_errors[k]

            k_gap = abs(
                k - k_oracle
            )

            error_gap = (
                test_error
                - oracle_error
            )

            print(
                f"{method:<20}"
                f"{k:<12}"
                f"{test_error:<15.4f}"
                f"{k_gap:<16}"
                f"{error_gap:<12.4f}"
            )

            result[method] = {
                "k": k,
                "test_error": test_error,
                "k_gap": k_gap,
                "error_gap": error_gap
            }

        results.append(
            result
        )

    return results


# ============================================================
# Decision-surface helper
# ============================================================

def make_grid(grid_size=100):
    """
    Create a grid over [0,1]^2 for decision-surface plots.
    """

    x1 = np.linspace(
        0,
        1,
        grid_size
    )

    x2 = np.linspace(
        0,
        1,
        grid_size
    )

    xx, yy = np.meshgrid(
        x1,
        x2
    )

    grid = np.column_stack([
        xx.ravel(),
        yy.ravel()
    ])

    return xx, yy, grid


# ============================================================
# Problem 5(b)
#
# Learning algorithm A:
# 5-fold CV chooses k.
# Then train k-NN on all S.
#
# ONE PNG:
#   - learning curve
#   - six decision surfaces
# ============================================================

def problem_5b(
    X_train,
    y_train,
    X_test,
    y_test
):
    selected_k = []
    test_errors = []

    models = []

    print()
    print("=" * 70)
    print("Problem 5(b): CV-Tuned k-NN")
    print("=" * 70)

    # --------------------------------------------------------
    # Train and evaluate
    # --------------------------------------------------------

    for m in M_VALUES:

        X_m = X_train[:m]
        y_m = y_train[:m]

        best_k = tune_k_cv(
            X_m,
            y_m
        )

        prediction = p4.predict_for_many_k(
            X_m,
            y_m,
            X_test,
            [best_k]
        )[best_k]

        test_error = error_rate(
            y_test,
            prediction
        )

        selected_k.append(
            best_k
        )

        test_errors.append(
            test_error
        )

        models.append(
            (
                X_m,
                y_m,
                best_k
            )
        )

        print(
            f"m = {m:4d} | "
            f"k = {best_k:2d} | "
            f"test error = {test_error:.4f}"
        )

    # ========================================================
    # One combined figure
    # ========================================================

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15, 13)
    )

    axes = axes.flatten()

    # --------------------------------------------------------
    # Learning curve
    # --------------------------------------------------------

    ax = axes[0]

    ax.plot(
        M_VALUES,
        test_errors,
        marker="o",
        label="CV-tuned k-NN"
    )

    ax.axhline(
        BAYES_ERROR,
        linestyle="--",
        label=f"Bayes error = {BAYES_ERROR:.3f}"
    )

    ax.set_xscale("log")

    ax.set_xlabel(
        "Training set size m"
    )

    ax.set_ylabel(
        "Test error rate"
    )

    ax.set_title(
        "Learning Curve"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    # --------------------------------------------------------
    # Decision surfaces
    # --------------------------------------------------------

    xx, yy, grid = make_grid(
        grid_size=100
    )

    for plot_idx, (
        m,
        model
    ) in enumerate(
        zip(M_VALUES, models),
        start=1
    ):

        X_m, y_m, best_k = model

        prediction = p4.predict_for_many_k(
            X_m,
            y_m,
            grid,
            [best_k]
        )[best_k]

        surface = prediction.reshape(
            xx.shape
        )

        ax = axes[plot_idx]

        ax.contourf(
            xx,
            yy,
            surface,
            alpha=0.3
        )

        ax.scatter(
            X_m[:, 0],
            X_m[:, 1],
            c=y_m,
            edgecolors="k",
            s=12
        )

        ax.set_title(
            f"m={m}, k={best_k}"
        )

        ax.set_xlabel(
            "$X_1$"
        )

        ax.set_ylabel(
            "$X_2$"
        )

        ax.set_xlim(
            0,
            1
        )

        ax.set_ylim(
            0,
            1
        )

    # Unused panels
    for idx in range(
        1 + len(M_VALUES),
        len(axes)
    ):
        axes[idx].axis(
            "off"
        )

    fig.suptitle(
        "Problem 5(b): 5-Fold CV Tuned k-NN",
        fontsize=16
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    plt.savefig(
        "problem_5b.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    return (
        selected_k,
        test_errors
    )


# ============================================================
# Problem 5(c)
#
# Decision Tree:
# tune max_depth with 5-fold CV.
# ============================================================

def tune_tree_depth_cv(
    X_train,
    y_train,
    num_folds=5,
    seed=0
):
    """
    Tune decision-tree max_depth using 5-fold CV.
    """

    rng = np.random.default_rng(
        seed
    )

    n = len(X_train)

    indices = rng.permutation(
        n
    )

    folds = np.array_split(
        indices,
        num_folds
    )

    errors = {
        depth: []
        for depth in DEPTH_VALUES
    }

    for fold_idx in range(
        num_folds
    ):

        val_idx = folds[
            fold_idx
        ]

        train_idx = np.concatenate([
            folds[j]
            for j in range(num_folds)
            if j != fold_idx
        ])

        X_fold_train = X_train[
            train_idx
        ]

        y_fold_train = y_train[
            train_idx
        ]

        X_val = X_train[
            val_idx
        ]

        y_val = y_train[
            val_idx
        ]

        for depth in DEPTH_VALUES:

            tree = DecisionTreeClassifier(
                max_depth=depth,
                random_state=0
            )

            tree.fit(
                X_fold_train,
                y_fold_train
            )

            prediction = tree.predict(
                X_val
            )

            errors[depth].append(
                error_rate(
                    y_val,
                    prediction
                )
            )

    mean_errors = {
        depth: np.mean(
            errors[depth]
        )
        for depth in DEPTH_VALUES
    }

    return choose_best(
        mean_errors
    )


def problem_5c(
    X_train,
    y_train,
    X_test,
    y_test
):
    knn_errors = []
    tree_errors = []

    knn_k_values = []
    tree_depth_values = []

    models = []

    print()
    print("=" * 80)
    print("Problem 5(c): k-NN vs Decision Tree")
    print("=" * 80)

    # --------------------------------------------------------
    # Train and evaluate both algorithms
    # --------------------------------------------------------

    for m in M_VALUES:

        X_m = X_train[:m]
        y_m = y_train[:m]

        # ====================================================
        # k-NN
        # ====================================================

        best_k = tune_k_cv(
            X_m,
            y_m
        )

        knn_prediction = p4.predict_for_many_k(
            X_m,
            y_m,
            X_test,
            [best_k]
        )[best_k]

        knn_error = error_rate(
            y_test,
            knn_prediction
        )

        # ====================================================
        # Decision Tree
        # ====================================================

        best_depth = tune_tree_depth_cv(
            X_m,
            y_m
        )

        tree = DecisionTreeClassifier(
            max_depth=best_depth,
            random_state=0
        )

        tree.fit(
            X_m,
            y_m
        )

        tree_prediction = tree.predict(
            X_test
        )

        tree_error = error_rate(
            y_test,
            tree_prediction
        )

        # ====================================================
        # Store results
        # ====================================================

        knn_k_values.append(
            best_k
        )

        tree_depth_values.append(
            best_depth
        )

        knn_errors.append(
            knn_error
        )

        tree_errors.append(
            tree_error
        )

        models.append(
            (
                X_m,
                y_m,
                best_k,
                tree,
                best_depth
            )
        )

        print(
            f"m = {m:4d} | "
            f"k-NN: k={best_k:2d}, "
            f"error={knn_error:.4f} | "
            f"Tree: depth={best_depth}, "
            f"error={tree_error:.4f}"
        )

    # ========================================================
    # ONE combined figure
    #
    # Row 0:
    #     learning curve
    #
    # Rows 1-3:
    #     kNN | Tree | kNN | Tree
    # ========================================================

    fig = plt.figure(
        figsize=(16, 17)
    )

    gs = fig.add_gridspec(
        4,
        4
    )

    # --------------------------------------------------------
    # Learning curve
    # --------------------------------------------------------

    ax_curve = fig.add_subplot(
        gs[0, :]
    )

    ax_curve.plot(
        M_VALUES,
        knn_errors,
        marker="o",
        label="k-NN"
    )

    ax_curve.plot(
        M_VALUES,
        tree_errors,
        marker="o",
        label="Decision Tree"
    )

    ax_curve.axhline(
        BAYES_ERROR,
        linestyle="--",
        label=f"Bayes error = {BAYES_ERROR:.3f}"
    )

    ax_curve.set_xscale(
        "log"
    )

    ax_curve.set_xlabel(
        "Training set size m"
    )

    ax_curve.set_ylabel(
        "Test error rate"
    )

    ax_curve.set_title(
        "Learning Curve: k-NN vs Decision Tree"
    )

    ax_curve.grid(
        True,
        alpha=0.3
    )

    ax_curve.legend()

    # --------------------------------------------------------
    # Grid used for all decision surfaces
    # --------------------------------------------------------

    xx, yy, grid = make_grid(
        grid_size=100
    )

    # --------------------------------------------------------
    # Decision surfaces
    # --------------------------------------------------------

    for idx, (
        m,
        model
    ) in enumerate(
        zip(M_VALUES, models)
    ):

        (
            X_m,
            y_m,
            best_k,
            tree,
            best_depth
        ) = model

        # Two values of m per row.
        row = (
            1 + idx // 2
        )

        # Each m uses two columns:
        # k-NN and tree.
        col = (
            idx % 2
        ) * 2

        # ====================================================
        # k-NN surface
        # ====================================================

        knn_prediction = p4.predict_for_many_k(
            X_m,
            y_m,
            grid,
            [best_k]
        )[best_k]

        knn_surface = knn_prediction.reshape(
            xx.shape
        )

        ax_knn = fig.add_subplot(
            gs[row, col]
        )

        ax_knn.contourf(
            xx,
            yy,
            knn_surface,
            alpha=0.3
        )

        ax_knn.scatter(
            X_m[:, 0],
            X_m[:, 1],
            c=y_m,
            edgecolors="k",
            s=10
        )

        ax_knn.set_title(
            f"k-NN: m={m}, k={best_k}"
        )

        ax_knn.set_xlabel(
            "$X_1$"
        )

        ax_knn.set_ylabel(
            "$X_2$"
        )

        ax_knn.set_xlim(
            0,
            1
        )

        ax_knn.set_ylim(
            0,
            1
        )

        # ====================================================
        # Decision-tree surface
        # ====================================================

        tree_prediction = tree.predict(
            grid
        )

        tree_surface = tree_prediction.reshape(
            xx.shape
        )

        ax_tree = fig.add_subplot(
            gs[row, col + 1]
        )

        ax_tree.contourf(
            xx,
            yy,
            tree_surface,
            alpha=0.3
        )

        ax_tree.scatter(
            X_m[:, 0],
            X_m[:, 1],
            c=y_m,
            edgecolors="k",
            s=10
        )

        ax_tree.set_title(
            f"Tree: m={m}, depth={best_depth}"
        )

        ax_tree.set_xlabel(
            "$X_1$"
        )

        ax_tree.set_ylabel(
            "$X_2$"
        )

        ax_tree.set_xlim(
            0,
            1
        )

        ax_tree.set_ylim(
            0,
            1
        )

    fig.suptitle(
        "Problem 5(c): k-NN vs Decision Tree",
        fontsize=17
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    plt.savefig(
        "problem_5c.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    return {
        "knn_k": knn_k_values,
        "knn_error": knn_errors,
        "tree_depth": tree_depth_values,
        "tree_error": tree_errors
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Use one RNG for drawing data.
    # --------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    # --------------------------------------------------------
    # Fixed test set
    #
    # Draw ONCE and reuse everywhere in Problem 5.
    # --------------------------------------------------------

    X_test, y_test = p4.draw_samples(
        10_000,
        rng
    )

    # --------------------------------------------------------
    # One trial:
    #
    # Draw S with 3000 examples once.
    # For each m, use the first m examples.
    # --------------------------------------------------------

    X_train, y_train = p4.draw_samples(
        3000,
        rng
    )

    # --------------------------------------------------------
    # Part (a)
    # --------------------------------------------------------

    results_5a = problem_5a(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Part (b)
    #
    # Saves:
    # problem_5b.png
    # --------------------------------------------------------

    selected_k, errors_5b = problem_5b(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Part (c)
    #
    # Saves:
    # problem_5c.png
    # --------------------------------------------------------

    results_5c = problem_5c(
        X_train,
        y_train,
        X_test,
        y_test
    )