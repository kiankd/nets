import lasagne
import numpy as np

"""
This script contains the default initialization parameters for a neural
network. It also builds a neural network based on parameters passed.

DEPRECATE - Theano is getting destroyed --> MUST SWITCH TO TENSOR FLOW
"""

DEFAULT_NEURAL_CONFIG = {
    'hidden_layers' : [100, 100, 100],
    'dropouts'      : [0.0, 0.0, 0.0, 0.0],
    'learning_rate' : 0.001,
    'lambda1'       : 0.0,
    'lambda2'       : 0.0,
    'rep_layers'    : [],
    'grad_descent'  : lasagne.updates.adam,
    'nonlinearity'  : lasagne.nonlinearities.rectify,
    'weights'       : lasagne.init.GlorotUniform,
    'random_seed'   : 1917,
}


def construct_neural_network(num_features, num_classes, config):
    """
    Constructs a neural network based on a configuration dictionary.
    :param num_features: int - designates the length of a sample's vector.
    :param num_classes: int - number of classes in train set.
    :param config: dictionary - designates neural net's configuration.
    :return: neural network
    """

    # set random seed before anything else
    rng = np.random.RandomState()
    rng.seed(config['random_seed'])
    lasagne.random.set_rng(rng)

    # syntactic sugar variables
    layers = config['hidden_layers']
    dropouts = config['dropouts']
    representation_layer_idxs = config['rep_layers']

    # Necessary assertion for allowing dropout to be included
    assert len(layers) + 1 == len(dropouts),\
        'Error in constructing neural network - there are {} layers designated'\
        ' but only {} dropout layers designated! There must be {} dropout' \
        ' layers designated!'.format(len(layers), len(dropouts), len(layers)+1)

    # Need to save list of layers we want to be representation layers.
    representation_layer_pointers = []

    # Initialize with the input layer!
    l_in = lasagne.layers.InputLayer(shape=(None, num_features))
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=dropouts[0])

    # Iterate over the number of layers designated
    crt_layer = l_in_drop
    for i in xrange(0, len(layers)):
        dropidx = i + 1
        hidden_layer = lasagne.layers.DenseLayer(
            crt_layer,
            num_units    = layers[i],
            nonlinearity = config['nonlinearity'],
            W            = config['weights'](),
        )

        # if designated as rep layer, add it!
        if i in representation_layer_idxs:
            representation_layer_pointers.append(hidden_layer)

        dropout_layer = lasagne.layers.DropoutLayer(hidden_layer,
                                                    p = dropouts[dropidx],
                                                    )
        crt_layer = dropout_layer

    # Now create the output layer and we're done!
    l_out = lasagne.layers.DenseLayer(
        crt_layer,
        num_units = num_classes,
        nonlinearity = lasagne.nonlinearities.softmax
    )

    return representation_layer_pointers, l_out
