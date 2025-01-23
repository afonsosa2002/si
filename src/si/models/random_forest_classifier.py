import numpy as np
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.data.dataset import Dataset
from si.base.model import Model

class RandomForestClassifier(Model):

    def __init__(self, 
                n_estimators: int = 100, 
                max_features: int = None, 
                min_sample_split: int = 2, 
                max_depth: int = 10, 
                mode: str = 'gini', 
                seed: int = 42, 
                 **kwargs): 
        """
        Initializes the Random Forest Classifier with specified hyperparameters.
        """

        super().__init__(**kwargs) 
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []  
    
    

    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Fits the Random Forest model to the dataset by training multiple decision trees.
        """
        np.random.seed(self.seed) 

        if self.max_features is None:
            self.max_features = int(np.sqrt(dataset.X.shape[1]))
        for _ in range(self.n_estimators): 

            bootstrap_indices = np.random.choice(dataset.X.shape[0], size=dataset.X.shape[0], replace=True)
            bootstrap_X = dataset.X[bootstrap_indices]
            bootstrap_y = dataset.y[bootstrap_indices]

            feature_indices = np.random.choice(dataset.X.shape[1], size=self.max_features, replace=False)
            bootstrap_features = [dataset.features[i] for i in feature_indices] 

            bootstrap_data = Dataset(X=bootstrap_X[:, feature_indices],
                                    y=bootstrap_y,
                                    features=bootstrap_features,
                                    label=dataset.label)

            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode
            )

            tree.fit(bootstrap_data)

            self.trees.append((tree, feature_indices)) 
        return self
    
    def _predict(self, dataset) -> np.ndarray:
        """
        Predicts target values for the given dataset using the ensemble of decision trees.
        """
        all_predictions = []

        for tree, features in self.trees: 
            
            if isinstance(features[0], int): 
                feature_indices = features
                feature_names = [dataset.features[i] for i in feature_indices]
            else:
                feature_names = features
                feature_indices = [dataset.features.index(f) if isinstance(f, str) else f for f in feature_names]

            X_subset = dataset.X[:, feature_indices] 
            tree_predictions = tree.predict(Dataset(X_subset, dataset.y, features=feature_names, label=dataset.label)) 
            all_predictions.append(tree_predictions) 
        
        all_predictions = np.array(all_predictions).T 
        final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=all_predictions)


        return np.array(final_predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculates the accuracy of the predictions compared to the true target values.
        """
        return accuracy(dataset.y, predictions) 