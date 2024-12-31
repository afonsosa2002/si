import numpy as np
from metrics import rmse

class KNNRegressor:
    def __init__(self, k=3, distance=None):
        self.k = k
        self.distance = distance if distance else self.euclidean_distance
        self.dataset = None

    def _fit(self, dataset):
        self.dataset = dataset
        return self

    def _predict(self, dataset):
        X_train, y_train = self.dataset
        X_test = dataset
        predictions = []

        for x_test in X_test:
            distances = np.array([self.distance(x_test, x_train) for x_train in X_train])
            nearest_indices = distances.argsort()[:self.k]
            nearest_values = y_train[nearest_indices]
            prediction = np.mean(nearest_values)
            predictions.append(prediction)

        return np.array(predictions)

    def _score(self, dataset):
        X_test, y_test = dataset
        y_pred = self._predict(X_test)
        return rmse(y_test, y_pred)