# Utility functions.
import numpy as np
import os
import errno
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def iter_accs():
    yield f1_score

def mse(true, pred):
    return np.mean((true - pred)**2)

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def results_write(fname, results_list):
    with open(fname, 'w') as f:
        for res_str in results_list:
            f.write(res_str)
            f.write('\n============================================\n\n')

def str_to_save_name(string):
    assert(len(string) < 100)
    return string.lower().replace(' ', '_')

def val_to_str(val):
    s = str(val)
    s = s.split('(')[0]
    s.replace(' ', '_')
    s.replace('.', 'e')
    s.replace('\n', '')
    return s

def params_to_str(params):
    return '_'.join(map(val_to_str, params.values()))


