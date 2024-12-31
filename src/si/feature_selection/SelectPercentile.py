import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.base import TransformerMixin

class SelectPercentile(TransformerMixin):
    def __init__(self, score_func=f_classif, percentile=10):
        self.score_func = score_func  
        self.percentile = percentile  
        
    def _fit(self, X, y):

        self.F, self.p = self.score_func(X, y)
        
        num_features = X.shape[1]
        num_selected = int(np.ceil(num_features * self.percentile / 100))
        
        top_features_idx = np.argsort(self.F)[::-1][:num_selected]
        
        self.selected_features = top_features_idx
        
        return self
    
    def _transform(self, X):
        return X[:, self.selected_features]