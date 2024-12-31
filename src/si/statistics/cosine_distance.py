import numpy as np

def cosine_distance(x, y):
    x = np.array(x)
    y = np.array(y)

    if x.ndim != 1 or y.ndim != 2 or x.shape[0] != y.shape[1]:
        return np.full(y.shape[0], np.nan)  

    dot_product = np.dot(y, x)

    x_norm = np.linalg.norm(x)

    y_norms = np.linalg.norm(y, axis=1)

    if x_norm == 0 or np.any(y_norms == 0):
        return np.full(y.shape[0], np.nan)

    cosine_similarity = dot_product / (x_norm * y_norms)
    return 1 - cosine_similarity