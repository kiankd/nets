import numpy as np

def clip_gradient(parameters, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in parameters:
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)
    return min(1, clip / (totalnorm + 1e-6))
