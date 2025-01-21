import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics import f_classification


class SelectPercentile(Transformer):
    
    def __init__(self, percentile:float, score_func:callable = f_classification,**kwargs):
        
        super().__init__(**kwargs)
        if isinstance(percentile,int):
            self.percentile = percentile
        else:
            raise ValueError("ERROR")
        self.score_func = score_func
        self.F = None
        self.p = None

    def _fit(self,dataset:Dataset) -> 'SelectPercentile':

        self.F,self.p = self.score_func(dataset) 
        return self
    
    def _transform(self, dataset: Dataset) -> Dataset:
    
        threshold= np.percentile(self.F,100-self.percentile)
        mask = self.F > threshold
        ties = np.where(self.F == threshold)[0]
        if len(ties) != 0:
            max_features = int (len(self.F)*self.percentile/100)
            mask[ties[: max_features -mask.sum()]] = True

        features = np.array(dataset.features)[mask]
        
        return Dataset(X=dataset.X[:, mask], y=dataset.y, features=list(features), label=dataset.label)
        