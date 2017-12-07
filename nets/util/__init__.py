# Utility functions.
import numpy as np
import os
import errno
import types
from copy import deepcopy
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

multi_class_f1 = lambda gold, pred: f1_score(gold, pred, average='weighted')

def iter_accs():
    # yield recall_score
    # yield precision_score
    yield 'F1 acc', multi_class_f1

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
    string = string.replace('.', 'e')
    return string.lower().replace(' ', '_')

def function_to_str(val):
    if type(val) == types.FunctionType:
        return str(deepcopy(val)())
    return val

def val_to_str(val):
    if type(val) is list:
        s = '-'.join(map(str, val))
    elif type(val) is dict:
        s = '-'.join(f'{k}_{v}' for k, v in val.items())
    elif type(val) == types.FunctionType:
        s = function_to_str(val)
    else:
        try:
            s = val.__name__
        except AttributeError:
            s = str(val)
    if not type(val) == types.FunctionType:
        s = s.split('(')[0]
    s.replace(' ', '_')
    s.replace('.', 'e')
    s.replace('\n', '')
    s.replace('[', '')
    s.replace(']', '')
    return s

def params_to_str(params):
    return '_'.join(map(val_to_str, params.values()))


