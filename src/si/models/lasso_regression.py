import numpy as np

class LassoRegression:
    def __init__(self, l1_penalty=0.1, scale=True):
        self.l1_penalty = l1_penalty
        self.scale = scale
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
    
    def _fit(self, X, y, max_iter=1000, patience=10):
        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X_scaled = (X - self.mean) / self.std
        else:
            X_scaled = X
        
        n_samples, n_features = X_scaled.shape
        self.theta = np.zeros(n_features)
        self.theta_zero = 0
        prev_cost = float('inf')
        for _ in range(max_iter):
            for j in range(n_features):
                residuals = y - self._predict(X_scaled)
                
                rho_j = np.dot(X_scaled[:, j], residuals)
                if rho_j < -self.l1_penalty:
                    self.theta[j] = (rho_j + self.l1_penalty) / n_samples
                elif rho_j > self.l1_penalty:
                    self.theta[j] = (rho_j - self.l1_penalty) / n_samples
                else:
                    self.theta[j] = 0
            
            self.theta_zero = np.mean(y - X_scaled.dot(self.theta))
            cost = self._compute_cost(X_scaled, y)
            if abs(prev_cost - cost) < 1e-6:
                break  
            prev_cost = cost
        
    def _predict(self, X):
        X_scaled = (X - self.mean) / self.std if self.scale else X
        return np.dot(X_scaled, self.theta) + self.theta_zero
    
    def _score(self, X, y):
        y_pred = self._predict(X)
        return np.mean((y - y_pred) ** 2)
    
    def _compute_cost(self, X, y):
        residuals = y - self._predict(X)
        cost = np.sum(residuals**2) / (2 * len(y))  
        l1_penalty = self.l1_penalty * np.sum(np.abs(self.theta))
        return cost + l1_penalty