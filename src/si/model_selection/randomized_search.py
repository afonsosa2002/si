import numpy as np

from model_selection.cross_validate import k_fold_cross_validation

def randomized_search_cv(model, X, y, hyperparameter_grid, scoring, cv=3, n_iter=10):
    # Dicionário para armazenar os resultados
    results = {'hyperparameters': [], 'scores': [], 'best_hyperparameters': None, 'best_score': -np.inf}

    # Verifica se o modelo possui todos os hiperparâmetros fornecidos
    for param in hyperparameter_grid:
        if not hasattr(model, param):
            raise ValueError(f"Model does not have hyperparameter '{param}'")

    all_combinations = []

    # Gera combinações aleatórias de hiperparâmetros
    for _ in range(n_iter):
        combination = {
            param: (
                np.random.choice(values)
                if isinstance(values[0], int)
                else np.random.uniform(values[0], values[1])
            )
            for param, values in hyperparameter_grid.items()
        }
        all_combinations.append(combination)

    # Avalia cada combinação de hiperparâmetros
    for combination in all_combinations:
        # Ajusta o modelo com os hiperparâmetros sorteados
        for param, value in combination.items():
            setattr(model, param, value)

        # Realiza a validação cruzada com a função k-fold 
        mean_score = k_fold_cross_validation(model, X, y, cv, scoring)  

        # Armazena a combinação de hiperparâmetros e o score obtido
        results['hyperparameters'].append(combination)
        results['scores'].append(mean_score)

        # Verifica se o score atual é o melhor e atualiza
        if mean_score > results['best_score']:
            results['best_score'] = mean_score
            results['best_hyperparameters'] = combination

    return results
