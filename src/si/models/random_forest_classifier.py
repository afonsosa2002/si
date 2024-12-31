import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2, max_depth=None, mode='gini', seed=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []  
    
    def _fit(self, X, y):
        np.random.seed(self.seed)  
        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))
        
        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            features = np.random.choice(X.shape[1], size=self.max_features, replace=False)
            X_bootstrap = X_bootstrap[:, features]
            
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, criterion=self.mode)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append((tree, features))
        return self
    
    def _predict(self, X):
        tree_preds = []
        for tree, features in self.trees:
            X_tree = X[:, features]
            tree_preds.append(tree.predict(X_tree))
        final_preds = []
        
        for i in range(len(X)):
            sample_preds = [pred[i] for pred in tree_preds]
            most_common = Counter(sample_preds).most_common(1)[0][0]
            final_preds.append(most_common)
        
        return np.array(final_preds)
    
    def _score(self, X, y):
        predictions = self._predict(X)
        return np.sum(predictions == y) / len(y)