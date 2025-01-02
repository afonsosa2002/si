import numpy as np
from unittest import TestCase
from si.model_selection import KNNRegressor
from si.metrics import rmse

class TestKNNRegressor(TestCase):

    def setUp(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y_train = np.array([1, 2, 3, 4, 5])
        self.X_train = X_train
        self.y_train = y_train
        self.dataset_train = (X_train, y_train)
        self.regressor = KNNRegressor(k=3)

    def test_init(self):
        self.assertEqual(self.regressor.k, 3)
        self.assertEqual(self.regressor.distance, self.regressor.euclidean_distance)

    def test_fit(self):
        self.regressor._fit(self.dataset_train)
        self.assertIsNotNone(self.regressor.dataset)

    def test_predict(self):
        self.regressor._fit(self.dataset_train)
        X_test = np.array([[2, 3], [6, 7]])
        predictions = self.regressor._predict(X_test)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(np.issubdtype(type(predictions[0]), np.floating))

    def test_score(self):
        self.regressor._fit(self.dataset_train)
        X_test = np.array([[2, 3], [6, 7]])
        y_test = np.array([1.5, 3.5])
        dataset_test = (X_test, y_test)
        score = self.regressor._score(dataset_test)
        self.assertTrue(score >= 0)
