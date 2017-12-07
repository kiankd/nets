from . import neural_model, basic_mlp, centroid_predictors
from torch import nn
from nets.util.constants import *


def get_default_params(input_dim, num_classes, layer_dims, lr):
    return {
        INPUT_DIM: input_dim,
        NUM_CLASSES: num_classes,
        DENSE_DIMS: layer_dims,
        ACTIVATION: nn.ReLU,
        WEIGHT_DECAY: 0,
        LOSS_FUN: nn.CrossEntropyLoss,
        LEARNING_RATE: lr,
        CLIP_NORM: 0,
    }

def make_mlp(name, unique_labels, params):
    print(params)

    # TODO: improve these silly "if" statements.
    if params[CORE]:
        if len(params[DENSE_DIMS]) == 1:
            mlp = centroid_predictors.CentroidSLP(params)
        elif len(params[DENSE_DIMS]) == 2:
            mlp = centroid_predictors.CentroidTLP(params)
        else:
            mlp = centroid_predictors.CentroidDNN(params)
    else:
        if len(params[DENSE_DIMS]) == 1:
            mlp = basic_mlp.SLP(params)
        elif len(params[DENSE_DIMS]) == 2:
            mlp = basic_mlp.TLP(params)
        else:
            mlp = basic_mlp.DMLP(params)

    model = neural_model.NeuralModel(name, mlp, params, unique_labels)
    return model
