import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n = 100

# Question 1
x1 = rng.multivariate_normal(
    mean=[0, 0],
    cov=[[1, 0], [0, 1]],
    size=n
)

# Question 2
x2 = rng.multivariate_normal(
    mean=[1, -1],
    cov=[[2, 0], [0, 2]],
    size=n
)

# Question 3
mu_a = [1, 0]
cov_a = [[1, 0.2],
         [0.2, 1]]

mu_b = [-1, 0]
cov_b = [[1, -0.2],
         [-0.2, 1]]

choose_a = rng.random(n) < 0.3

x3 = np.empty((n, 2))
x3[choose_a] = rng.multivariate_normal(
    mu_a, cov_a, choose_a.sum()
)
x3[~choose_a] = rng.multivariate_normal(
    mu_b, cov_b, (~choose_a).sum()
)


fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)

datasets = [
    (x1, "Question 1"),
    (x2, "Question 2"),
    (x3, "Question 3")
]

for ax, (data, title) in zip(axes, datasets):
    ax.scatter(data[:, 0], data[:, 1])
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("hw0.png")