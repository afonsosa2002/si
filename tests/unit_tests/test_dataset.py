from typing import Self
import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())
        
    def setUp(self):
        X = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9], [np.nan, 11, 12]])
        y = np.array([1, 2, 3, 4])
        features = np.array(['a', 'b', 'c'])
        label = 'y'
        self.obj = Dataset(X, y, features, label)

    def test_dropna(self):
        self.obj.dropna()
        expected_X = np.array([[7, 8, 9]])
        expected_y = np.array([3])
        np.testing.assert_array_equal(self.obj.X, expected_X)
        np.testing.assert_array_equal(self.obj.y, expected_y)

    def test_fillna_mean(self):
        self.obj.fillna('mean')
        expected_X = np.array([[1, 2, 6], [4, 7, 6], [7, 8, 9], [4, 11, 9]])
        np.testing.assert_array_equal(self.obj.X, expected_X)

    def test_fillna_median(self):
        self.obj.fillna('median')
        expected_X = np.array([[1, 2, 9], [4, 8, 9], [7, 8, 9], [4, 11, 9]])
        np.testing.assert_array_equal(self.obj.X, expected_X)

    def test_fillna_value(self):
        self.obj.fillna(0)
        expected_X = np.array([[1, 2, 0], [4, 0, 6], [7, 8, 9], [0, 11, 12]])
        np.testing.assert_array_equal(self.obj.X, expected_X)

    def test_remove_by_index(self):
        self.obj.remove_by_index([1, 3])
        expected_X = np.array([[1, 2, np.nan], [7, 8, 9]])
        expected_y = np.array([1, 3])
        np.testing.assert_array_equal(self.obj.X, expected_X)
        np.testing.assert_array_equal(self.obj.y, expected_y)