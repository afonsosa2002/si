import numpy as np
from unittest import TestCase
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from si.model_selection import StackingClassifier

class TestStackingClassifier(TestCase):

    def setUp(self):
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        model_1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model_2 = LogisticRegression(max_iter=200)
        self.stack_model = StackingClassifier(models=[model_1, model_2], final_model=LogisticRegression())

    def test_init(self):
        self.assertEqual(len(self.stack_model.models), 2)
        self.assertIsInstance(self.stack_model.final_model, LogisticRegression)

    def test_fit(self):
        self.stack_model._fit(self.X_train, self.y_train)
        self.assertTrue(len(self.stack_model.models_) > 0)

    def test_predict(self):
        self.stack_model._fit(self.X_train, self.y_train)
        predictions = self.stack_model._predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertTrue(np.issubdtype(type(predictions[0]), np.integer))

    def test_score(self):
        self.stack_model._fit(self.X_train, self.y_train)
        score = self.stack_model._score(self.X_test, self.y_test)
        self.assertGreater(score, 0.9)