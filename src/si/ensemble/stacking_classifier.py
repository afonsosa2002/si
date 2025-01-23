from typing import List
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier(Model):

    def __init__(self, models: List[Model], final_model: Model, **kwargs):
        """
        Initializes the StackingClassifier with base models and a final model.
        """
        super().__init__(**kwargs)
        self.models = models           
        self.final_model = final_model 
        
        self.predictions_dataset = None

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Trains the base models and the final model on the provided dataset.
        """
        for model in self.models:
            model.fit(dataset)

        base_predictions = np.column_stack([model.predict(dataset) for model in self.models])

        self.predictions_dataset = Dataset(X=base_predictions, y=dataset.y, label=dataset.label)

        self.final_model.fit(self.predictions_dataset)

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Generates predictions using the base models and the final model.
        """
        base_predictions = np.column_stack([model.predict(dataset) for model in self.models])
        predictions = self.final_model.predict(Dataset(X=base_predictions))

        return predictions


    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Computes the accuracy of the predictions against the true labels.
        """
        return accuracy(dataset.y, predictions)