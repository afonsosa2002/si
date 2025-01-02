import numpy as np
from unittest import TestCase
from si.metrics import rmse

class TestRMSE(TestCase):

    def test_rmse_perfect_prediction(self):
        y_true = [3, 2, 1]
        y_pred = [3, 2, 1]
        result = rmse(y_true, y_pred)
        self.assertEqual(result, 0)