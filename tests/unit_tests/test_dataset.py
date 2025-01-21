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
        
    def test_dropna(self):
        X = np.array([[1, 2, 3], [np.nan, 5, 6], [7, np.nan, 9], [10, 11, 12]])
        y = np.array([1, 2, 3, 4])
        features = ['a', 'b', 'c']
        label = 'y'

        dataset = Dataset(X, y, features, label)                    
        dataset.dropna()                                            

        self.assertEqual(dataset.shape(), (2, 3))                                   
        np.testing.assert_array_equal(dataset.y, np.array([1, 4]))  


    def test_fillna(self):
        
        X = np.array([[1,       2, np.nan], 
                    [4,  np.nan,      6], 
                    [7,       8,      9], 
                    [np.nan, 11,    12]])
        y = np.array([1, 2, 3, 4])
        dataset = Dataset(X.copy(), y)

        dataset.fillna(0.0)
        expected_X_value = np.array([[1, 2,   0], 
                                    [4, 0,   6], 
                                    [7, 8,   9], 
                                    [0, 11, 12]])
        self.assertFalse(np.isnan(dataset.X).any())
        np.testing.assert_array_equal(dataset.X, expected_X_value)

    def test_remove_by_index(self):
        
        X = np.array([[1, 2, 3], 
                    [4, 5, 6], 
                    [7, 8, 9]])
        y = np.array([1, 2, 3])
        dataset = Dataset(X, y)

        dataset.remove_by_index(0)  
        expected_X_after_second_removal = np.array([[7, 8, 9]])
        expected_y_after_second_removal = np.array([3])
        
        np.testing.assert_array_equal(dataset.X, expected_X_after_second_removal)
        np.testing.assert_array_equal(dataset.y, expected_y_after_second_removal)