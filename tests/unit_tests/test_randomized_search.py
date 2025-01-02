import numpy as np
from unittest import TestCase
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from model_selection import randomized_search_cv


class TestRandomizedSearchCV(TestCase):

    def setUp(self):
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.model = LogisticRegression(max_iter=200)

    def test_randomized_search_cv_with_valid_input(self):
        hyperparameter_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2']
        }
        scoring = accuracy_score
        results = randomized_search_cv(self.model, self.X, self.y, hyperparameter_grid, scoring, cv=3, n_iter=5)

        self.assertTrue('hyperparameters' in results)
        self.assertTrue('scores' in results)
        self.assertTrue('best_hyperparameters' in results)
        self.assertTrue('best_score' in results)

        self.assertGreater(results['best_score'], 0)  
        self.assertIsNotNone(results['best_hyperparameters'])  

    def test_randomized_search_cv_with_invalid_model(self):
        with self.assertRaises(ValueError):
            hyperparameter_grid = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2']
            }
            model = object()  
            randomized_search_cv(model, self.X, self.y, hyperparameter_grid, accuracy_score, cv=3, n_iter=5)