# Utility functions.
import numpy as np
import os
import errno

def mse(true, pred):
    return np.mean((true - pred)**2)

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
