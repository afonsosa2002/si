import numpy as np
from unittest import TestCase
from sklearn.datasets import load_iris
from si.model_selection import RandomForestClassifier

class TestRandomForestClassifier(TestCase):

    def setUp(self):
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.rf = RandomForestClassifier(n_estimators=10, max_features=2, mode='gini', seed=42)

    def test_init(self):
        self.assertEqual(self.rf.n_estimators, 10)
        self.assertEqual(self.rf.max_features, 2)
        self.assertEqual(self.rf.mode, 'gini')

    def test_fit(self):
        self.rf._fit(self.X, self.y)
        self.assertGreater(len(self.rf.trees), 0)

    def test_predict(self):
        self.rf._fit(self.X, self.y)
        X_test = self.X[:5]
        predictions = self.rf._predict(X_test)
        self.assertEqual(len(predictions), 5)
        self.assertTrue(np.issubdtype(type(predictions[0]), np.integer))

    def test_score(self):
        self.rf._fit(self.X, self.y)
        score = self.rf._score(self.X, self.y)
        self.assertGreater(score, 0.9)