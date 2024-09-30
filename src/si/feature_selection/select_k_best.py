from si.base.estimator import Estimator
from si.base.transformer import Transformer
from si.data.dataset import Dataset
import nunpy

class SelectBest(Transformer):
    
    def __init__(self, score_func: callable,k: int,**kwargs ):
        super().__init__(**kwargs)
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None
        
    def _fit(self, dataset: Dataset) -> 'SelectBest':
        
        self.F, self.p = self.score_func(dataset)
        
        return self

    def _transform(self,dataset: Dataset) -> Dataset:
        idx = np.argsort(self.F)
        mask = idx[-self.k:]
        new_X = self.dataset.X[:, mask]
        new_features = self.dataset.features = new_features, y=dataset.y, label=dataset.label)
        
        return

