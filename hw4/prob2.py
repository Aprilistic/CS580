# Recommended imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load a subset of MNIST (e.g., 10,000 samples for fast computation)
print("Loading MNIST data...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data[:10000].astype(np.float64)
y = mnist.target[:10000].astype(int)
print(f"Data shape: {X.shape}, Labels shape: {y.shape}")


class MyPCA:
    def __init__(self, n_components: int = 2):
        """
        Parameters:
        -----------
        n_components : int
            Number of principal components to retain (k).
        """
        self.n_components = n_components
        self.mean_ = None              # Shape: (D,)
        self.components_ = None        # Shape: (k, D) - each row is an eigenvector u_j^T
        self.eigenvalues_ = None       # Shape: (D,) - sorted eigenvalues
        self.explained_variance_ratio_ = None # Shape: (D,) - proportion of variance

    def fit(self, X: np.ndarray):
        """
        Fit the PCA model on data matrix X of shape (N, D).
        """
        N, D = X.shape
        # Normalization
        X = X / 255.0

        # 1. Compute and store empirical mean (shape: (D,))
        self.mean_ = np.mean(X, axis=0)

        plt.imshow(self.mean_.reshape(28, 28), cmap='gray')
        plt.savefig('mean.png')

        # 2. Center the data
        X_tilde = X - self.mean_

        # 3. Compute the D x D covariance matrix
        cov_matrix = (X_tilde.T @ X_tilde) / N

        # 4. Compute eigenvalues and eigenvectors using np.linalg.eigh
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 5. Sort eigenvalues and eigenvectors in DESCENDING order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.eigenvalues_ = eigenvalues

        # 6. Compute explained variance ratio: lambda_j / sum(all lambdas)
        self.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)

        # 7. Store the top n_components eigenvectors as rows in self.components_
        # (shape: (n_components, D))
        self.components_ = eigenvectors[:, :self.n_components].T

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X into the k-dimensional latent space.
        Returns Z of shape (N, n_components).
        """
        # 1. Normalize
        X = X / 255.0

        # 2. Center using the mean computed during fit()
        X_tilde = X - self.mean_

        # 3. Project onto principal components
        Z = X_tilde @ self.components_.T

        return Z

    # def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
    #     """
    #     Reconstruct original data from low-dimensional representation Z.
    #     Returns X_reconstructed of shape (N, D).
    #     """
    #     # YOUR CODE HERE
    #     pass



# Unit Test: Orthogonality & Reconstruction Dimensions
pca_test = MyPCA(n_components=5)
pca_test.fit(X)

# Test 1: Orthonormality of components (U^T @ U should be identity matrix)
U = pca_test.components_.T # shape: (D, k)
ortho_error = np.max(np.abs(U.T @ U - np.eye(5)))
assert ortho_error < 1e-10, f"Components are not orthonormal! Error: {ortho_error}"

# Test 2: Transform and inverse transform shapes
Z_test = pca_test.transform(X[:10])
assert Z_test.shape == (10, 5), f"Expected shape (10, 5), got {Z_test.shape}"

# X_rec_test = pca_test.inverse_transform(Z_test)
# assert X_rec_test.shape == (10, 784), f"Expected shape (10, 784), got {X_rec_test.shape}"

print("All sanity checks passed successfully!")


# Q3
pca = MyPCA(n_components=2)
pca.fit(X)

Z = pca.transform(X)

print(Z.shape)  # (10000, 2)

plt.figure(figsize=(9, 7))

for digit in range(10):
    mask = (y == digit)

    plt.scatter(
        Z[mask, 0],
        Z[mask, 1],
        label=str(digit),
        s=5,
        alpha=0.5,
        color=plt.cm.tab10(digit)
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Digit")
plt.savefig("pca_2d.png", dpi=300)
plt.show()

# Q4
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for i in range(2):
    component = pca.components_[i].reshape(28, 28)

    vmax = np.max(np.abs(component))

    im = axes[i].imshow(
        component,
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax
    )

    axes[i].set_title(f"Principal Component {i+1}")
    axes[i].axis("off")

    fig.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.savefig("eigen_digits.png", dpi=300)
plt.show()