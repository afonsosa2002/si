import numpy as np
import os
from unittest import TestCase
from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.statistics.cosine_distance import cosine_distance

class TestCosineDistance(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_cosine_distance(self):
        x = np.array([1, 2, 3])
        y = np.array([[1, 0, 0], [0, 1, 0], [1, 2, 3]])

        computed_distance = cosine_distance(x, y)

        manual_distance = 1 - np.array([
            np.dot(x, y[0]) / (np.linalg.norm(x) * np.linalg.norm(y[0])),
            np.dot(x, y[1]) / (np.linalg.norm(x) * np.linalg.norm(y[1])),
            np.dot(x, y[2]) / (np.linalg.norm(x) * np.linalg.norm(y[2]))
        ])

        # Verifica se as distâncias estão próximas
        assert np.allclose(computed_distance, manual_distance), f"Computed: {computed_distance}, Manual: {manual_distance}"

