from unittest import TestCase

from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv
import numpy as np
from si.model_selection.split import stratified_train_test_split, train_test_split

class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)
        
    def test_stratified_train_test_split(self):
        train, test = stratified_train_test_split(self.dataset, test_size=0.33, random_state=42)
        
        self.assertEqual(train.shape()[0], 4)
        self.assertEqual(test.shape()[0], 2)

        train_labels = train.y
        test_labels = test.y
        
        unique_train, counts_train = np.unique(train_labels, return_counts=True)
        unique_test, counts_test = np.unique(test_labels, return_counts=True)
        
        self.assertTrue(np.array_equal(unique_train, [0, 1]))
        self.assertTrue(np.array_equal(unique_test, [0, 1]))
        self.assertEqual(counts_train[0], 2)
        self.assertEqual(counts_train[1], 2)
        self.assertEqual(counts_test[0], 1)
        self.assertEqual(counts_test[1], 1)