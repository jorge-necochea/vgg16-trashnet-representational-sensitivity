import numpy as np
from .. import T
class Dataset: pass
def _collate(items):
    if isinstance(items[0], tuple):
        return tuple(_collate(list(z)) for z in zip(*items))
    return np.stack([np.asarray(x) for x in items]).view(T)
class DataLoader:
    def __init__(self, ds, batch_size=1, num_workers=0, **k): self.ds, self.bs = ds, batch_size
    def __iter__(self):
        for s in range(0, len(self.ds), self.bs):
            yield _collate([self.ds[i] for i in range(s, min(s + self.bs, len(self.ds)))])
