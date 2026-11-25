# Sustituto mínimo de torch para probar en CPU la lógica sin red (sin GPU ni torch real).
import numpy as np
class T(np.ndarray):
    def numpy(self): return np.asarray(self)
    def item(self): return np.asarray(self).item()
def tensor(x, dtype=None): return np.asarray(x, dtype=np.float64 if dtype is not None else None).view(T)
def from_numpy(a): return np.asarray(a).view(T)
float64 = 'float64'; float32 = 'float32'
__version__ = 'stub'
