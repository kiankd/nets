from nets import AbstractModel
from collections import OrderedDict
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Wrapper for our RNN.
class WinnogradRNNModel(AbstractModel):
    def __init__(self, name, vocabulary, embedding_dim, encoder_hidden_dim, dense_hidden_dim,
                 learning_rate=0.01, hyperparameters=None, pretrained_emb=None):
        super(WinnogradRNNModel, self).__init__(name, hyperparameters)
        self.model = WinnogradRNN(
            embedding_dim=embedding_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            dense_hidden_dim=dense_hidden_dim,
            vocab_size=len(vocabulary),
            pretrained_emb=pretrained_emb,
        )

        # this indexes all of the words, something torch always uses for embeddings
        self.word_to_idx = {}
        for word in vocabulary:
            if not self.word_to_idx.has_key(word):
                self.word_to_idx[word] = len(self.word_to_idx)

        # init loss function and optimizer
        self.loss_function = nn.SmoothL1Loss() #seems better than MSE #nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    # see torch documentation for explanation of why we do this in this way
    # noinspection PyArgumentList
    def prepare_sequence(self, word_seq):
        idxs = [self.word_to_idx[word] for word in word_seq]
        tensor = torch.LongTensor(idxs)
        return autograd.Variable(tensor)

    def predict(self, x):
        super(WinnogradRNNModel, self).predict(x)
        self.model.eval()

    def train(self, x, y, reset=True):
        """
        This function will train our RNN model. Note that the x input will not be
        a list of float vectors, but a list of 2-tuples. Each tuple has two lists
        of preprocessed text
            row k: ('ESUBJ refused EOBJ a permit'.split(),
                    'because E* feared violence.'.split())
                Where E* = ESUBJ for row k; E* = EOBJ for row k+1.
        We are assuming that this is called each time a sample is to be propagated
        and then backpropagated through; e.g., it receives as input a single sample.

        TODO: allow for mini-batch input! Now it only takes one sample at a time!
        """
        super(WinnogradRNNModel, self).train(x, y)
        # Step 1, clear out the history of the gradient from previous sample,
        # should always be doing this for each sample/MB.
        if reset:
            self.model.zero_grad()
            self.model.init_hiddens()

        # Step 2, prepare variables input. Here we are assuming x is a mini-batch!
        # seqs = [self.prepare_sequence(wseq) for wseq in x]
        sample = autograd.Variable(self.prepare_sequence(x))
        target = autograd.Variable(y)

        # Step 3, feed forward, then backprop!
        likelihood_prediction = self.model.forward(sample)
        loss = self.loss_function(likelihood_prediction, target)
        loss.backward()
        self.optimizer.step()

        # Might need to use gradient clipping... saving code here for later
        # torch.nn.utils.clip_grad_norm(self.model.parameters(), clip_size)
        # for p in self.model.parameters():
        #     p.data.add_(-lr, p.grad.data)

# Global model.
class WinnogradRNN(nn.Module):
    def __init__(self, embedding_dim, encoder_hidden_dim, dense_hidden_dim, vocab_size,
                 dropout=0.25, pretrained_emb=None):
        """
        :param embedding_dim: size of embeddings
        :param encoder_hidden_dim: size of hidden layer in our encoders
        :param dense_hidden_dim: size of our dense combination layer
        :param vocab_size: size of our vocabularity
        :param pretrained_emb: numpy array of shape (vocab_size, embedding_dim)
        """
        super(WinnogradRNN, self).__init__()

        # initialize word embedding table
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_emb is not None:
            assert(pretrained_emb.shape == (vocab_size, embedding_dim,))
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_emb))

        # initialize our encoders
        self.drop = nn.Dropout(p=dropout)
        self.context_encoder = EncoderRNN(embedding_dim, encoder_hidden_dim)
        self.query_encoder = EncoderRNN(embedding_dim, encoder_hidden_dim)

        # initialize our dense combiner layer, may need to add multiple hidden layers in future
        dense_input_size = 4 * encoder_hidden_dim # because 4 hidden states are given
        self.combiner_output = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(dense_input_size, dense_hidden_dim)),
            ('relu1', nn.ReLU()),
            ('drop-global1', self.drop), # dropout from main dense layer
            ('dense2', nn.Linear(dense_hidden_dim, 1,)), # one output
            ('sigmoid-likelihood', nn.Sigmoid()),
        ]))

    def forward(self, sentence):
        """
        :param sentence: tensor of word-idx sequences
        :return: scalar likelihood prediction, confidence of correct entity placement
        """
        seq_len = len(sentence)
        word_embs = self.drop(self.word_embeddings(sentence)) #dropout embeds
        context_emb = self.context_encoder.forward(seq_len, word_embs.view(seq_len, 1, -1))
        query_emb = self.query_encoder.forward(seq_len, word_embs.view(seq_len, 1, -1))
        concated_repr = self.drop(torch.cat(context_emb, query_emb)) #dropout reprs
        # TODO: ensure this is concatenating on correct dim!!
        return self.combiner_output(concated_repr)

    def init_hiddens(self):
        self.context_encoder.init_hiddens()
        self.query_encoder.init_hiddens()


# RNN encoder for sequence - the global model feeds it everything it needs to know.
class EncoderRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(EncoderRNN, self).__init__()

        # initialize hidden state h0, diff for forward and backward
        self.hidden_dim = hidden_dim
        self.hidden_f, self.hidden_b = None, None
        self.init_hiddens()

        # initialize RNNs - two separate ones learnt for forward and backward
        self.rnn_f = nn.GRUCell(embedding_dim, hidden_dim) # forward rnn
        self.rnn_b = nn.GRUCell(embedding_dim, hidden_dim) # backward rnn

    def init_hiddens(self):
        self.hidden_f = self._build_init_hidden()
        self.hidden_b = self._build_init_hidden()

    def _build_init_hidden(self):
        return autograd.Variable(torch.zeros(1, 1, self.hidden_dim))

    def forward(self, seq_length, embedding_seq):
        """
        :param seq_length: int, size of the given sequence
        :param embedding_seq: tensor of a sequence of embeddings
        :return: returns an embedding of a sequence by using bidirectional GRU and
                 concatenating the final hidden layers obtained from each direction.
        """
        hf = self.hidden_f
        hb = self.hidden_b
        for i in range(seq_length):
            hf = self.rnn_f(embedding_seq[i], hf) # start at beginning of seq
            hb = self.rnn_b(embedding_seq[seq_length-i-1], hb) # "" end of seq
        return torch.cat([hf, hb], 1)
