from nets import AbstractModel
from collections import OrderedDict
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Wrapper for our RNN.
class WinnogradRNNModel(AbstractModel):
    def __init__(self, name, hyperparameters=None):
        super(WinnogradRNNModel, self).__init__(name, hyperparameters)

    def predict(self, x):
        super(WinnogradRNNModel, self).predict(x)

    def train(self, x, y):
        super(WinnogradRNNModel, self).train(x, y)


# Global model.
class WinnogradRNN(nn.Module):
    def __init__(self, embedding_dim, encoder_hidden_dim, dense_hidden_dim, vocab_size, pretrained_emb=None):
        """
        :param embedding_dim:
        :param hidden_dim:
        :param vocab_size:
        :param pretrained_emb: numpy array of shape (vocab_size, embedding_dim)
        """
        super(WinnogradRNN, self).__init__()

        # initialize word embedding table
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_emb is not None:
            assert(pretrained_emb.shape == (vocab_size, embedding_dim,))
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_emb))

        # initialize our encoders
        self.context_encoder = EncoderRNN(embedding_dim, encoder_hidden_dim)
        self.query_encoder = EncoderRNN(embedding_dim, encoder_hidden_dim)

        # initialize our dense combiner layer, may need to add multiple hidden layers in future
        dense_input_size = 4 * encoder_hidden_dim # because 4 hidden states are given
        self.combiner_output = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(dense_input_size, dense_hidden_dim)),
            ('tanh1', nn.Tanh()),
            ('dense2', nn.Linear(dense_hidden_dim, 1)),
            ('sigmoid-likelihood', nn.Sigmoid()),
        ]))

    def forward(self, sentence):
        """
        :param sentence: tensor of word-idx sequences
        :return: scalar likelihood prediction, confidence of correct entity placement
        """
        seq_len = len(sentence)
        word_embs = self.word_embeddings(sentence)
        context_emb = self.context_encoder.forward(seq_len, word_embs.view(seq_len, 1, -1))
        query_emb = self.query_encoder.forward(seq_len, word_embs.view(seq_len, 1, -1))
        concated_repr = torch.cat(context_emb, query_emb)
        # TODO: ensure this is concatenating on correct dim!!
        return self.combiner_output(concated_repr)


# RNN encoder for sequence - the global model feeds it everything it needs to know.
class EncoderRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(EncoderRNN, self).__init__()

        # initialize hidden state h0, diff for forward and backward
        self.hidden_dim = hidden_dim
        self.hidden_forward = self.init_hidden()
        self.hidden_back = self.init_hidden()

        # initialize RNN
        self.rnn = nn.GRUCell(embedding_dim, hidden_dim)

    def init_hidden(self):
        return autograd.Variable(torch.zeros(1, 1, self.hidden_dim))

    def forward(self, seq_length, embedding_seq):
        """
        :param seq_length: int, size of the given sequence
        :param embedding_seq: tensor of a sequence of embeddings
        :return: returns an embedding of a sequence by using bidirectional GRU and
        concatenating the final hidden layers obtained from each direction.
        """
        hf = self.hidden_forward
        hb = self.hidden_back
        for i in range(seq_length):
            hf = self.rnn(embedding_seq[i], hf) # start at beginning of seq
            hb = self.rnn(embedding_seq[seq_length-i-1], hb) # start at end of seq

        # TODO: ensure this is concatenating on correct dim!!
        return torch.cat(hf, hb)
