from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.models.random_forest_classifier import RandomForestClassifier
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier

class TestRandomForest(TestCase):

    def setUp(self):

        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):
        """
        Testa o método _fit para garantir que os parâmetros de inicialização
        sejam corretamente atribuídos e as árvores sejam treinadas.
        """
        random_forest = RandomForestClassifier(n_estimators=10, max_depth=10, min_sample_split=5, mode='gini') 

        self.assertEqual(random_forest.min_sample_split, 5) 
        self.assertEqual(random_forest.max_depth, 10) 
        self.assertEqual(len(random_forest.trees), 10)  
    def test_predict(self):

        random_forest = RandomForestClassifier(n_estimators=10, max_depth=10, min_sample_split=5, mode='gini')
        random_forest.fit(self.train_dataset) 

        predictions = random_forest.predict(self.test_dataset)  

        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])

    def test_score(self):

        random_forest = RandomForestClassifier(n_estimators=10, max_depth=10, min_sample_split=5, mode='gini')
        random_forest.fit(self.train_dataset)
        
        accuracy_ = random_forest.score(self.test_dataset) 
        self.assertEqual(round(accuracy_, 2), 0.95) 