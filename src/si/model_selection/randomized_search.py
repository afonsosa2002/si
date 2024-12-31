import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score  

def k_fold_cross_validation(model, X, y, cv, scoring):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores.append(scoring(y_test, predictions))

    return np.mean(scores)

def randomized_search_cv(model, X, y, hyperparameter_grid, scoring, cv=3, n_iter=10):
    results = {'hyperparameters': [],'scores': [],'best_hyperparameters': None,'best_score': -np.inf }
    for param in hyperparameter_grid:
        if not hasattr(model, param):
            raise ValueError(f"Model does not have hyperparameter '{param}'")

    all_combinations = []
    
    for _ in range(n_iter):
        combination = {param: np.random.choice(values) for param, values in hyperparameter_grid.items()}
        all_combinations.append(combination)

    for combination in all_combinations:
        for param, value in combination.items():
            setattr(model, param, value)
            
        mean_score = k_fold_cross_validation(model, X, y, cv, scoring)
        results['hyperparameters'].append(combination)
        results['scores'].append(mean_score)
        
        if mean_score > results['best_score']:
            results['best_score'] = mean_score
            results['best_hyperparameters'] = combination
            
    return results
