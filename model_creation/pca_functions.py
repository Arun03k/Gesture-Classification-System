"""
PCA_Functions.py — Manual PCA implementation (NumPy only, no sklearn).
Used for O2: Dimensionality reduction.
"""

import numpy as np
import pandas as pd


class ManualPCA:
    """
    PCA via eigendecomposition of the covariance matrix.
    NumPy only — no sklearn.
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None            # (n_features, n_components)
        self.eigenvalues = None
        self.explained_variance_ratio = None
        self.pca_feature_names = None

    def fit(self, X):
        """Fit PCA on training data X of shape (n_samples, n_features)."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float64)

        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Eigen decomposition (eigh for symmetric matrices)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort descending
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Select components
        self.components = eigenvectors[:, :self.n_components]
        self.eigenvalues = eigenvalues[:self.n_components]

        total_var = eigenvalues.sum()
        self.explained_variance_ratio = self.eigenvalues / (total_var + 1e-15)
        self.pca_feature_names = [f"PC{i+1}" for i in range(self.n_components)]

        return self

    def transform(self, X):
        """Project X onto principal components: (n, d) -> (n, k)."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


# ---- Standalone functions (compatible with reference project API) ----

def fit_pca(X, n_components=48, save_path=None):
    """
    Fit PCA and optionally save parameters.
    Returns (X_pca, pca_feature_names).
    """
    pca = ManualPCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    if save_path is not None:
        np.savez(
            save_path,
            mean=pca.mean,
            components=pca.components,
            eigenvalues=pca.eigenvalues,
            explained_variance_ratio=pca.explained_variance_ratio,
        )
        print(f"PCA params saved to: {save_path}")

    return X_pca, pca.pca_feature_names


def transform_pca(X, load_path):
    """
    Transform X using saved PCA parameters.
    Returns DataFrame with PC columns.
    """
    params = np.load(load_path)
    mean = params["mean"]
    components = params["components"]

    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.asarray(X, dtype=np.float64)
    X_centered = X - mean
    X_pca = np.dot(X_centered, components)

    n_components = components.shape[1]
    feature_names = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(X_pca, columns=feature_names)
