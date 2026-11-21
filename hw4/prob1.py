
import numpy as np
import pandas as pd
from collections import Counter

# ==========================================
# Dataset
# ==========================================

X = [
    ["money", "gift", "money"],
    ["money", "friend", "card"],
    ["money", "due", "gift"],
    ["friend", "happy", "friend"],
]

Y = ["spam", "spam", "spam", "not-spam"]


# ==========================================
# Q1. Prior Probabilities
# ==========================================

def compute_priors(Y):
    priors = {}

    classes = np.unique(Y)
    for cls in classes:
        count = Y.count(cls)
        priors[cls] = count / len(Y)

    return priors


# ==========================================
# Q2. Vocabulary
# ==========================================

def build_vocabulary(X):
    vocabulary = np.unique(X)

    return vocabulary


# ==========================================
# Q3. Conditional Probabilities
# ==========================================

def compute_likelihoods(X, Y, vocabulary):
    """Return add-one smoothed word probabilities for each class."""
    counts_by_class = {label: Counter() for label in np.unique(Y)}
    for document, label in zip(X, Y):
        counts_by_class[label].update(document)

    likelihoods = {}
    vocabulary_size = len(vocabulary)
    for label, word_counts in counts_by_class.items():
        smoothed_total = sum(word_counts.values()) + vocabulary_size
        likelihoods[label] = {
            word: (word_counts[word] + 1) / smoothed_total for word in vocabulary
        }

    return likelihoods


# ==========================================
# Q4. Laplace Smoothing
# ==========================================

# TODO: Explain why smoothing is useful when
# P(money | not-spam) is calculated.


# ==========================================
# Q5. Posterior Classification
# ==========================================

X_test = ["friend", "happy", "friend", "due"]

def predict(X_test, priors, likelihoods, classes):
    scores = {}

    for cls in classes:
        score = priors[cls]
        class_likelihoods = likelihoods[cls]
        for word in X_test:
            score *= class_likelihoods[word]
        scores[cls] = score

    print("Scores:", scores)
    return max(scores, key=scores.get)


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    priors = compute_priors(Y)
    vocabulary = build_vocabulary(X)
    likelihoods = compute_likelihoods(X, Y, vocabulary)

    print("Priors:", priors)
    print("Vocabulary:", vocabulary)
    print("Vocab size:", len(vocabulary))
    likelihoods_df = pd.DataFrame(likelihoods)
    print("Likelihoods:", likelihoods_df)

    prediction = predict(X_test, priors, likelihoods, np.unique(Y))
    print("Prediction:", prediction)
