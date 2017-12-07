import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.util.constants import *
from abc import ABCMeta, abstractmethod


class CentroidNN(nn.Module):
    """
    Abstract class for dealing with centroid-based prediction.
    """
    __metaclass__ = ABCMeta

    def __init__(self, params):
        super(CentroidNN, self).__init__()
        self.params = params
        self.centroid_matrix = None

    def update_centroids(self, centroid_matrix):
        """
        We store the centroid matrix and have to normalize it to make inference.
        :param centroid_matrix: this is a tensor in the GPU.
                It is of size K by H, where K is the number of classes, and
                H is the dimensionality of our latent representations.
        :return: None
        """
        self.centroid_matrix = F.normalize(centroid_matrix, p=2, dim=1)

    def get_centroid_predictions(self, latent_rep_matrix):
        dots = torch.mm(latent_rep_matrix, self.centroid_matrix.transpose(0, 1))
        return dots # returns the confidence scores for predictions of each possible class

    def needs_centroids(self):
        return self.centroid_matrix is None

    def forward(self, x):
        reps = self.get_reps(x)
        return self.get_centroid_predictions(reps), reps

    @abstractmethod
    def get_reps(self, x):
        """
        :param x: samples matrix.
        :return: latent representations of the samples, however the model seeks to build them.
        """
        pass

# Specific MLP models.
class CentroidSLP(CentroidNN):
    """
    Single-layer perceptron.
    """
    def __init__(self, params):
        super(CentroidSLP, self).__init__(params)
        self.input_to_h = nn.Linear(self.params[INPUT_DIM], self.params[DENSE_DIMS][0])
        self.activation = self.params[ACTIVATION]()

    def get_reps(self, x):
        super(CentroidSLP, self).get_reps(x)
        return self.activation(self.input_to_h(x))


class CentroidTLP(CentroidNN):
    """
    Two-layer perceptron.
    """
    def __init__(self, params):
        super(CentroidTLP, self).__init__(params)
        self.input_to_h1 = nn.Linear(self.params[INPUT_DIM], self.params[DENSE_DIMS][0])
        self.activation1 = self.params[ACTIVATION]()
        self.h1_to_h2 = nn.Linear(self.params[DENSE_DIMS][0], self.params[DENSE_DIMS][1])
        self.activation2 = self.params[ACTIVATION]()

    def get_reps(self, x):
        super(CentroidTLP, self).get_reps(x)
        h1_repr = self.activation1(self.input_to_h1(x))
        return self.activation2(self.h1_to_h2(h1_repr))


class CentroidDNN(CentroidNN):
    """
    N-layer perceptron.
    """
    def __init__(self, params):
        super(CentroidDNN, self).__init__(params)
        input_to_h1 = nn.Linear(self.params[INPUT_DIM], self.params[DENSE_DIMS][0])
        activation1 = self.params[ACTIVATION]()
        sequence = [input_to_h1, activation1]
        prev_dim = self.params[DENSE_DIMS][0]
        for next_dim in self.params[DENSE_DIMS][1:]:
            sequence.append(nn.Linear(prev_dim, next_dim))
            sequence.append(self.params[ACTIVATION]())
            prev_dim = next_dim
        self.all_layers = nn.Sequential(*sequence)

    def get_reps(self, x):
        super(CentroidDNN, self).get_reps(x)
        return self.all_layers(x)
