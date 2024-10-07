import numpy as np

def train_test_split(dataset, test_size= float, random_state=123):
    np.random.seed(random_state)
    
    permutations = np.random.permutation(dataset.X.shape()[0])
    
    test_sample_size = int(dataset.shape()[0] + test_size)
    
    test_idx = permutations[:test_sample_size]
    train_idx = permutations[test_sample_size]
    
    train_dataset = dataset(X=dataset.X[train_indices], y=dataset.y[train_indices])
    test_dataset = dataset(X=dataset.X[test_indices], y=dataset.y[test_indices])
    
    return train_set, test_set