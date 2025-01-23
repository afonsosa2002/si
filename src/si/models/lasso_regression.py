import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse

class LassoRegression(Model):

    def __init__(self, l1_penalty: float = 1.0, 
                scale: bool = True, 
                max_iter: int = 1000, 
                patience: int = 5, 
                tolerance: float = 1e-4, **kwargs):
        """        
        Initializes the Lasso Regression model with specified hyperparameters.
        """
        super().__init__(**kwargs)
        self.l1_penalty = l1_penalty
        self.scale = scale
        self.max_iter = max_iter
        self.patience = patience
        self.tolerance = tolerance  

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> 'LassoRegression':
        """
        Fits the Lasso model to the dataset.
        """
        if self.scale:
            
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
        else:
            
            X = dataset.X

        y = dataset.y
        n_features = X.shape[1]

        self.theta = np.zeros(n_features)
        self.theta_zero = 0

        early_stopping = 0
        for iter in range(self.max_iter):   
            theta_prev = self.theta.copy()  

            for j in range(n_features):     
            
                residuals = y - (X.dot(self.theta) - self.theta[j] * X[:, j])
                rho_j = X[:, j].T.dot(residuals)

                if rho_j > self.l1_penalty:
                    self.theta[j] = (rho_j - self.l1_penalty) / np.sum(X[:, j] ** 2)    
                elif rho_j < -self.l1_penalty:
                    self.theta[j] = (rho_j + self.l1_penalty) / np.sum(X[:, j] ** 2)    
                else:
                    self.theta[j] = 0                                                   

            self.theta_zero = np.mean(y - X.dot(self.theta))

            if np.linalg.norm(self.theta - theta_prev, ord=1) < self.tolerance:
                early_stopping += 1
            else:
                early_stopping = 0

            if early_stopping >= self.patience:
                break                               

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts target values using the fitted Lasso model.
        """
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        predictions = X.dot(self.theta) + self.theta_zero

        return predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """        
        Calculates the mean squared error (MSE) between true and predicted values.
        """
        return mse(dataset.y, predictions)