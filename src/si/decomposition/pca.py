import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class PCA(Transformer):

    def __init__(self, n_components):
        """
        Initializes the PCA transformer with the specified number of components.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, dataset: Dataset):
        """
        Computes the principal components and explained variance from the dataset.
        """
        X=dataset.X
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        self.components = sorted_eigenvectors[:, :self.n_components]

        total_variance = np.sum(sorted_eigenvalues)
        self.explained_variance = sorted_eigenvalues[:self.n_components] / total_variance

    def _transform(self, dataset):
        """
        Projects the dataset onto the principal components.
        """
        X = dataset.X
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)