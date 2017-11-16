import torch
import torch.nn as nn
from collections import OrderedDict

INPUT_DIM = 'input_dim'
DENSE_DIMS = 'dense_layers'
ACTIVATION = 'activation'
NUM_CLASSES = 'num_classes'

class MLP(nn.Module):
    def __init__(self, params):
        super(MLP, self).__init__()
        self.params = params

        layers = [('dense1', nn.Linear(params[INPUT_DIM], params[DENSE_DIMS][0])),
                  ('nonlin1', params[ACTIVATION]())]
        for i in range(1, len(params[DENSE_DIMS])):
            layers.append(
                ('dense{}'.format(i+1),
                nn.Linear(params[DENSE_DIMS][i-1], params[DENSE_DIMS][i]))
            )
            layers.append(
                ('nonlin{}'.format(i+1),
                 params[ACTIVATION]()) # initiate activation function
            )

        layers.append(
            ('dense_end',
             nn.Linear(params[DENSE_DIMS][-1], params[NUM_CLASSES]))
        )
        layers.append(('softmax', nn.Softmax()))

        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.layers(x)

