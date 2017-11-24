import torch
import torch.nn as nn
from collections import OrderedDict

INPUT_DIM = 'input_dim'
DENSE_DIMS = 'dense_layers'
ACTIVATION = 'activation'
NUM_CLASSES = 'num_classes'
REPRESENTATION_LAYER = 'rep_layer'

class MLP(nn.Module):
    def __init__(self, params):
        super(MLP, self).__init__()
        self.params = params

        # set to last hidden layer if not specified
        if not REPRESENTATION_LAYER in self.params:
            self.params[REPRESENTATION_LAYER] = len(self.params[DENSE_DIMS])-1

        # First hidden layer (assuming we have at least 1 hidden layer).
        layers = [nn.Linear(self.params[INPUT_DIM], self.params[DENSE_DIMS][0]),
                  self.params[ACTIVATION]()]
        if len(self.params[DENSE_DIMS]) == 1:
            self.params[REPRESENTATION_LAYER] = len(layers)

        # Iterate to make an arbitrary number of layers (probs will only be 2 or 3).
        for i in range(1, len(self.params[DENSE_DIMS])):
            layers.append(
                nn.Linear(self.params[DENSE_DIMS][i-1], self.params[DENSE_DIMS][i])
            )
            layers.append(
                 self.params[ACTIVATION]() # initiate activation function
            )

            # Below we are indexing our representation layer, this is because
            # the index that is passed won't correspond directly to the sequential
            # layers ordering since nonlinearities and input count as layers.
            if i == self.params[REPRESENTATION_LAYER]:
                self.params[REPRESENTATION_LAYER] = len(layers)

        # Output layer, softmax.
        layers.append(
             nn.Linear(self.params[DENSE_DIMS][-1], self.params[NUM_CLASSES])
        )
        layers.append(nn.Softmax())

        # Construct the final sequence.
        self.layers = nn.Sequential(*layers)

    def forward(self, x, get_rep=False):
        if get_rep:
            output = self.layers(x) # feed forward
            cut_model = nn.Sequential(*list(self.layers.children())[:self.params[REPRESENTATION_LAYER]])
            return output, cut_model(x)
        return self.layers(x)

