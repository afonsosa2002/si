import numpy as np
from unittest import TestCase
from sklearn.datasets import make_classification
from si.decomposition.pca import PCA

class TestPCA(TestCase):

    def setUp(self):
        X, _ = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
        self.X = X
        self.pca = PCA(n_components=2)

    def test_init(self):
        self.assertEqual(self.pca.n_components, 2)
        self.assertIsNone(self.pca.mean)
        self.assertIsNone(self.pca.components)
        self.assertIsNone(self.pca.explained_variance)

    def test_fit(self):
        self.pca._fit(self.X)
        self.assertIsNotNone(self.pca.mean)
        self.assertIsNotNone(self.pca.components)
        self.assertIsNotNone(self.pca.explained_variance)
        self.assertEqual(self.pca.components.shape[1], 2)

    def test_transform(self):
        self.pca._fit(self.X)
        X_transformed = self.pca._transform(self.X)
        self.assertEqual(X_transformed.shape[1], 2)