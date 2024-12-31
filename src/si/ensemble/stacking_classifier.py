import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, models, final_model):
        self.models = models
        self.final_model = final_model
    
    def _fit(self, X, y):
        self.models_ = []
        for model in self.models:
            model.fit(X, y)
            self.models_.append(model)
        
        predictions = [model.predict(X) for model in self.models_]
        stacked_predictions = np.column_stack(predictions)
        self.final_model.fit(stacked_predictions, y)
        return self
    
    def _predict(self, X):
        predictions = [model.predict(X) for model in self.models_]
        stacked_predictions = np.column_stack(predictions)
        return self.final_model.predict(stacked_predictions)
    
    def _score(self, X, y):
        predictions = self._predict(X)
        return accuracy_score(y, predictions)