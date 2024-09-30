from unittest import TestCase
from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv



class TestKmeans(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
