from si.data.dataset import Dataset
import scipy

def f_classification(dataset: Dataset) -> tuple:
    classes = dataset.get_classes()
    
    groups = []
    for class in classes:
        mask= dataset.y == class
        
        group = dataset.X[mask . :]
        group.append(group)
        
    return scipy.stats.f_oneway(*groups)
        
    
    
    