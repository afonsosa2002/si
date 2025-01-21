from typing import Callable, Tuple, Dict, Any
import numpy as np
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation

def randomized_search(model, dataset: Dataset, hyperparameter_grid: Dict[str, Tuple], scoring: Callable = None, cv: int = 5, n_iter: int = None) -> Dict[str, Any]:
    
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    param_names = list(hyperparameter_grid.keys())
    
    all_combinations = []
    for _ in range(n_iter):
        combination = {}
        for param_name in param_names:
            param_values = hyperparameter_grid[param_name]
            random_value = np.random.choice(param_values)
            combination[param_name] = random_value
        all_combinations.append(combination)

    all_scores = []
    
    for params in all_combinations:
        
        for param_name, param_value in params.items():
            setattr(model, param_name, param_value)
        
        fold_scores = k_fold_cross_validation(model, dataset, scoring, cv)
        
        mean_score = np.mean(fold_scores)
        all_scores.append(mean_score)
    
    best_idx = np.argmax(all_scores)
    best_score = all_scores[best_idx]
    best_hyperparameters = all_combinations[best_idx]

    return {
        'hyperparameters': all_combinations,
        'scores': all_scores,
        'best_hyperparameters': best_hyperparameters,
        'best_score': best_score
    }

if __name__ == '__main__':
    from si.data.dataset import Dataset
    from si.models.logistic_regression import LogisticRegression
    from si.model_selection.split import train_test_split
    from si.io.csv_file import read_csv

    data = read_csv('datasets/breast_bin/breast-bin.csv', sep=',', features=True, label=True) 

    train, test = train_test_split(data, test_size=0.33, random_state=42)

    lr = LogisticRegression()

    parameter_grid_ = {
        'l2_penalty': np.linspace(1, 10, 10),
        'alpha': np.linspace(0.001, 0.0001, 100),
        'max_iter': np.linspace(1000, 2000, 200)
    }

    results_ = randomized_search(model=lr,
                                    dataset=data,
                                    hyperparameter_grid=parameter_grid_,
                                    cv=3,
                                    n_iter=10)

    hyperparameters = results_['hyperparameters']
    print(f"Hyperparameters: {hyperparameters}") 

    best_hyperparameters = results_['best_hyperparameters']
    print(f"Best hyperparameters: {best_hyperparameters}") 
    
    scores = results_['scores']
    print(f"Scores: {scores}")

    best_score = results_['best_score']
    print(f"Best score: {best_score}")