import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())