import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from nets import AbstractModel
from nets.model.neural.neural_model import NeuralModel
from collections import OrderedDict, Counter
from itertools import izip
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

INPUT_DIM = 'input_dim'
DENSE_DIMS = 'dense_layers'
ACTIVATION = 'activation'
NUM_CLASSES = 'num_classes'


class MLP(nn.Module):
    def __init__(self, params):
        super(MLP, self).__init__()
        self.params = params

        layers = [('dense1', nn.Linear(params[INPUT_DIM], params[DENSE_DIMS][0]))]
        for i in range(1, len(params[DENSE_DIMS])):
            layers.append(
                ('nonlin{}'.format(i),
                 params[ACTIVATION]()) # initiate activation function
            )
            layers.append(
                ('dense{}'.format(i+1),
                nn.Linear(params[DENSE_DIMS][i-1], params[DENSE_DIMS][i]))
            )
        layers.append(
            ('dense_end',
             nn.Linear(params[DENSE_DIMS][-1], params[NUM_CLASSES]))
        )
        layers.append(('softmax', nn.Softmax()))

        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.layers(x)

