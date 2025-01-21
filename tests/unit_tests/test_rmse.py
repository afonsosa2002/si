import numpy as np
from unittest import TestCase
from sklearn.metrics import mean_squared_error
from src.si.metrics.rmse import rmse

class TestRMSE(TestCase):

    def test_rmse(self):
        
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])

        custom_rmse = rmse(y_true, y_pred)

        sklearn_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        self.assertAlmostEqual(custom_rmse, sklearn_rmse)