from . import neural_model, basic_mlp
from torch import nn


def get_default_params(input_dim, num_classes, layer_dims, lr):
    return {
        basic_mlp.INPUT_DIM: input_dim,
        basic_mlp.NUM_CLASSES: num_classes,
        basic_mlp.DENSE_DIMS: layer_dims,
        basic_mlp.ACTIVATION: nn.ReLU,
        neural_model.WEIGHT_DECAY: 0,
        neural_model.LOSS_FUN: nn.CrossEntropyLoss,
        neural_model.LEARNING_RATE: lr,
        neural_model.CLIP_NORM: 0,
    }

def make_mlp(name, unique_labels, params):
    print(params)
    mlp = basic_mlp.MLP(params)
    model = neural_model.NeuralModel(name, mlp, params, unique_labels)
    return model
