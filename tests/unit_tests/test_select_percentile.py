import numpy as np
from unittest import TestCase
from sklearn.datasets import make_classification
from si.feature_selection.select_percentile import SelectPercentile

class TestSelectPercentile(TestCase):

    def setUp(self):
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_classes=2, random_state=42)
        self.X = X
        self.y = y
        self.selector = SelectPercentile(score_func=None, percentile=50)

    def test_init(self):
        self.assertEqual(self.selector.score_func, None)
        self.assertEqual(self.selector.percentile, 50)

    def test_fit(self):
        self.selector._fit(self.X, self.y)
        self.assertTrue(hasattr(self.selector, 'selected_features'))
        num_selected = int(np.ceil(self.X.shape[1] * self.selector.percentile / 100))
        self.assertEqual(len(self.selector.selected_features), num_selected)

    def test_transform(self):
        self.selector._fit(self.X, self.y)
        X_transformed = self.selector._transform(self.X)
        num_selected = int(np.ceil(self.X.shape[1] * self.selector.percentile / 100))
        self.assertEqual(X_transformed.shape[1], num_selected)
        selected_columns = self.selector.selected_features
        self.assertTrue(np.array_equal(X_transformed, self.X[:, selected_columns]))