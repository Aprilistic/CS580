"""CSC 480/580 Homework 2, Problem 1.

Part (a) prints the separate 95% confidence intervals and overlap result.
Part (b) performs the paired t-test on the evaluator-level differences.

Run with: uv run --with numpy --with scipy python p1.py
"""

# %% Problem 1: Paired t-test

import numpy as np
from scipy.stats import t

A_RATINGS = np.array([3, 4, 3, 2, 4, 5, 4, 2, 3, 4, 1, 2], dtype=float)
B_RATINGS = np.array([1, 3, 3, 2, 2, 3, 5, 2, 2, 1, 2, 1], dtype=float)


def mean_confidence_interval(ratings, confidence=0.95):
    """Return sample mean, sample SD, t critical value, and confidence interval."""
    ratings = np.asarray(ratings, dtype=float)
    n = ratings.size
    if n < 2:
        raise ValueError("At least two ratings are required.")
    sample_mean = np.mean(ratings)
    sample_sd = np.std(ratings, ddof=1)  # Uses n - 1 in the denominator.
    standard_error = sample_sd / np.sqrt(n)
    critical_value = t.ppf((1 + confidence) / 2, df=n - 1)
    margin = critical_value * standard_error
    return sample_mean, sample_sd, critical_value, (sample_mean - margin, sample_mean + margin)


def problem_1a():
    """Compute a 95% confidence interval for each algorithm's mean rating."""
    a_mean, a_sd, a_critical, a_ci = mean_confidence_interval(A_RATINGS)
    b_mean, b_sd, b_critical, b_ci = mean_confidence_interval(B_RATINGS)

    print(f"t critical values: A={a_critical:.4f}, B={b_critical:.4f}")
    print(
        f"A: mean={a_mean:.4f}, sample SD={a_sd:.4f}, "
        f"95% CI=({a_ci[0]:.4f}, {a_ci[1]:.4f})"
    )
    print(
        f"B: mean={b_mean:.4f}, sample SD={b_sd:.4f}, "
        f"95% CI=({b_ci[0]:.4f}, {b_ci[1]:.4f})"
    )
    intervals_overlap = max(a_ci[0], b_ci[0]) <= min(a_ci[1], b_ci[1])
    print(f"Confidence intervals overlap: {intervals_overlap}")

    return a_ci, b_ci, intervals_overlap


def problem_1b():
    """Perform the two-sided paired t-test at alpha = 0.05."""
    alpha = 0.05
    differences = B_RATINGS - A_RATINGS
    n = differences.size
    degrees_of_freedom = n - 1
    mean_difference = np.mean(differences)
    sample_sd = np.std(differences, ddof=1)
    standard_error = sample_sd / np.sqrt(n)

    # Observed statistic: how many standard errors the sample mean is from 0.
    t_statistic = mean_difference / standard_error

    # Critical cutoff: 2.5% of the t distribution lies above this value.
    t_critical = t.ppf(1 - alpha / 2, df=degrees_of_freedom)
    margin = t_critical * standard_error
    confidence_interval = (mean_difference - margin, mean_difference + margin)

    # Two-sided probability of a t statistic at least this extreme under H0.
    p_value = 2 * t.sf(abs(t_statistic), df=degrees_of_freedom)
    reject_null = p_value < alpha

    print(f"Differences (B - A): {differences.astype(int).tolist()}")
    print(f"Mean difference: {mean_difference:.4f}")
    print(f"Sample SD of differences: {sample_sd:.4f}")
    print(f"Standard error: {standard_error:.4f}")
    print(f"Observed t statistic: {t_statistic:.4f}")
    print(f"Critical t value for 95% CI: +/-{t_critical:.4f}")
    print(f"Two-sided p-value: {p_value:.4f}")
    print(
        "95% CI for mean difference: "
        f"({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})"
    )
    print(f"Reject H0 at alpha=0.05: {reject_null}")

    return t_statistic, p_value, confidence_interval


if __name__ == "__main__":
    print("Problem 1(a)")
    problem_1a()
    print("\nProblem 1(b)")
    problem_1b()
