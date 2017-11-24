import torch
import torch.nn as nn
from collections import OrderedDict

INPUT_DIM = 'input_dim'
DENSE_DIMS = 'dense_layers'
ACTIVATION = 'activation'
NUM_CLASSES = 'num_classes'
REPRESENTATION_LAYER = 'rep_layer'

def has_core(param_d):
    return len(param_d['core']) > 0

class SLP(nn.Module):
    """
    Single-layer perceptron.
    """
    def __init__(self, params):
        super(SLP, self).__init__()
        self.params = params
        self.input_to_h = nn.Linear(self.params[INPUT_DIM], self.params[DENSE_DIMS][0])
        self.activation1 = self.params[ACTIVATION]()
        self.h_to_output = nn.Linear(self.params[DENSE_DIMS][0], self.params[NUM_CLASSES])
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        h1_repr = self.activation1(self.input_to_h(x))
        if has_core(self.params):
            predictions = self.softmax(self.h_to_output(h1_repr.detach()))
        else:
            predictions = self.softmax(self.h_to_output(h1_repr))
        return predictions, h1_repr


class TLP(nn.Module):
    """
    Two-layer perceptron.
    """
    def __init__(self, params):
        super(TLP, self).__init__()
        self.params = params
        self.input_to_h1 = nn.Linear(self.params[INPUT_DIM], self.params[DENSE_DIMS][0])
        self.activation1 = self.params[ACTIVATION]()
        self.h1_to_h2 = nn.Linear(self.params[DENSE_DIMS][0], self.params[DENSE_DIMS][1])
        self.activation2 = self.params[ACTIVATION]()
        self.h2_to_output = nn.Linear(self.params[DENSE_DIMS][1], self.params[NUM_CLASSES])
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        h1_repr = self.activation1(self.input_to_h1(x))
        if has_core(self.params):
            h2_repr = self.activation2(self.h1_to_h2(h1_repr.detach()))
        else:
            h2_repr = self.activation2(self.h1_to_h2(h1_repr))
        predictions = self.softmax(self.h2_to_output(h2_repr))
        return predictions, h1_repr
