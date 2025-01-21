from typing import Callable, Union

import numpy as np

from src.si.base.model import Model
from src.si.data.dataset import Dataset
from src.si.metrics.rmse import rmse
from src.si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):

        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':

        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:

        distances = self.distance(sample, self.dataset.X)
        k_nearest_neighbors = np.argsort(distances)[:self.k]
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        return np.mean(k_nearest_neighbors_labels)

    def _predict(self, dataset: Dataset) -> np.ndarray:

        predictions = np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)
        return predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        
        return rmse(dataset.y, predictions)