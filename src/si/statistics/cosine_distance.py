import numpy as np

def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the cosine distance between a vector and a set of vectors.
    """
    dot_product = np.dot(y, x)          
    norm_x = np.linalg.norm(x)          
    norm_y = np.linalg.norm(y, axis=1) 
    similarity = dot_product / (norm_x * norm_y)
    return 1 - similarity