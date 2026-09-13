### p3.py 

from dataclasses import dataclass
import math

# ============================================================
# Dataset
# ============================================================

# y -> 1
# n -> 0
#
# Rating:
#   +2, +1, 0 -> positive label (1)
#   -1, -2    -> negative label (0)

FEATURES = ["Easy", "AI", "Sys", "Thy", "Morning"]

data = [
    # rating, Easy, AI, Sys, Thy, Morning
    (+2, 1, 1, 0, 1, 0),
    (+2, 1, 1, 0, 1, 0),
    (+2, 0, 1, 0, 0, 0),
    (+2, 0, 0, 0, 1, 0),
    (+2, 0, 1, 1, 0, 1),

    (+1, 1, 1, 0, 0, 0),
    (+1, 1, 1, 0, 1, 0),
    (+1, 0, 1, 0, 1, 0),

    ( 0, 0, 0, 0, 0, 1),
    ( 0, 1, 0, 0, 1, 1),
    ( 0, 0, 1, 0, 1, 0),
    ( 0, 1, 1, 1, 1, 1),

    (-1, 1, 1, 1, 0, 1),
    (-1, 0, 0, 1, 1, 0),
    (-1, 0, 0, 1, 0, 1),
    (-1, 1, 0, 1, 0, 1),

    (-2, 0, 0, 1, 1, 0),
    (-2, 0, 1, 1, 0, 1),
    (-2, 1, 0, 1, 0, 0),
    (-2, 1, 0, 1, 0, 1),
]


# Convert into a more convenient representation
dataset = []

for rating, easy, ai, sys, thy, morning in data:
    dataset.append({
        "Easy": easy,
        "AI": ai,
        "Sys": sys,
        "Thy": thy,
        "Morning": morning,
        "label": 1 if rating >= 0 else 0
    })


# ============================================================
# Tree structure
# ============================================================

@dataclass
class Node:
    feature: str | None = None

    # Child when feature == 0
    left: "Node | None" = None

    # Child when feature == 1
    right: "Node | None" = None

    # Only used for leaf nodes
    label: int | None = None

    # You can store entropy here for reporting
    entropy: float | None = None

    def is_leaf(self):
        return self.label is not None


# ============================================================
# Helper functions
# ============================================================

def entropy(rows):
    """
    Calculate entropy of the labels in rows.

    H(Y) = - s p(y)
    """
    n = len(rows)

    if n == 0:
        return 0

    positive = sum(row["label"] == 1 for row in rows)
    negative = sum(row["label"] == 0 for row in rows)

    p_pos = positive / n
    p_neg = negative / n

    ent = 0.0

    if p_pos > 0:
        ent -= p_pos * math.log2(p_pos)

    if p_neg > 0:
        ent -= p_neg * math.log2(p_neg)

    return ent


def majority_label(rows):
    """
    Return the most common label (0 or 1) in rows.
    """
    positive = sum(row["label"] == 1 for row in rows)
    negative = sum(row["label"] == 0 for row in rows)
    return 1 if positive >= negative else 0


def split_data(rows, feature):
    """
    Split rows based on a binary feature.

    Returns:
        no_rows  -> feature == 0
        yes_rows -> feature == 1
    """
    no_rows = [row for row in rows if row[feature] == 0]
    yes_rows = [row for row in rows if row[feature] == 1]
    return no_rows, yes_rows


def split_entropy(rows, feature):
    """
    Calculate the weighted entropy after splitting on feature.

    H_after =
        |NO| / |data| * H(NO)
        +
        |YES| / |data| * H(YES)

    Smaller is better.
    """
    n = len(rows)
    no_rows, yes_rows = split_data(rows, feature)
    return (
        len(no_rows) / n * entropy(no_rows)
        + len(yes_rows) / n * entropy(yes_rows)
    )


def best_feature(rows, remaining_features):
    """
    Find the feature producing the lowest weighted entropy.

    Returns:
        best_feature_name
    """
    best_feat = None
    min_entropy = float("inf")

    for f in remaining_features:
        se = split_entropy(rows, f)
        if se < min_entropy:
            min_entropy = se
            best_feat = f

    return best_feat


# ============================================================
# Decision tree training
# ============================================================

def decision_tree_train(
    rows,
    remaining_features,
    max_depth,
    depth=0
):
    # Most frequent label
    guess = majority_label(rows)

    # Current node entropy
    current_entropy = entropy(rows)

    # Base case 1: labels are unambiguous
    labels = [row["label"] for row in rows]
    if len(set(labels)) == 1:
        return Node(
            label=guess,
            entropy=current_entropy
        )

    # Base case 2: no features remain
    if len(remaining_features) == 0:
        return Node(
            label=guess,
            entropy=current_entropy
        )

    # Base case 3: maximum depth reached
    if depth == max_depth:
        return Node(
            label=guess,
            entropy=current_entropy
        )

    # Choose feature with smallest split entropy
    best_feat = best_feature(rows, remaining_features)

    # Split dataset
    no_rows, yes_rows = split_data(rows, best_feat)

    # Make a NEW feature list
    new_features = [
        f for f in remaining_features
        if f != best_feat
    ]

    # Recursively construct children
    left = decision_tree_train(
        no_rows,
        new_features.copy(),
        max_depth,
        depth + 1
    )

    right = decision_tree_train(
        yes_rows,
        new_features.copy(),
        max_depth,
        depth + 1
    )

    return Node(
        feature=best_feat,
        left=left,
        right=right,
        entropy=current_entropy
    )


# ============================================================
# Prediction
# ============================================================

def decision_tree_test(tree, test_point):
    """
    Predict the label of one example.
    """

    # Base case: leaf node
    if tree.is_leaf():
        return tree.label

    # Follow the branch according to the feature value
    if test_point[tree.feature] == 0:
        return decision_tree_test(tree.left, test_point)
    else:
        return decision_tree_test(tree.right, test_point)



# ============================================================
# Display tree
# ============================================================

def print_tree(tree, depth=0, branch="Root"):
    indent = "    " * depth

    if tree.is_leaf():
        print(f"{indent}{branch} -> Leaf: {tree.label}")
        return

    print(
        f"{indent}{branch} -> {tree.feature} "
        f"(Entropy: {tree.entropy:.3f})"
    )

    print_tree(
        tree.left,
        depth + 1,
        branch="No (0)"
    )

    print_tree(
        tree.right,
        depth + 1,
        branch="Yes (1)"
    )

# ============================================================
# Train
# ============================================================

tree = decision_tree_train(
    dataset,
    remaining_features=FEATURES.copy(),
    max_depth=2
)

print_tree(tree)