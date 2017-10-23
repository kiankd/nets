from nets import AbstractModel
from collections import OrderedDict

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np

np.random.seed(1)
def random_word_emb(dim):
    return 0.2 * np.random.uniform(-1.0, 1.0, dim)

# Wrapper class over our abstract model paradigm.
class SentiBiRNN(AbstractModel):
    UNK = '*UNKNOWN*'

    def __init__(self, name, vocabulary, embedding_dim, encoder_hidden_dim, dense_hidden_dim,
                 labels, dropout, learning_rate=0.01, hyperparameters=None, emb_dict=None):

        super(SentiBiRNN, self).__init__(name, hyperparameters)

        # this indexes all of the words, something torch always uses for embeddings
        self.word_to_idx = {}
        embeddings = []
        for word in vocabulary:
            if not self.word_to_idx.has_key(word):
                self.word_to_idx[word] = len(self.word_to_idx)
                embeddings.append(emb_dict[word])

        # don't forget unk!
        self.word_to_idx[self.UNK] = len(self.word_to_idx)
        embeddings.append(random_word_emb(embedding_dim))
        embeddings = np.array(embeddings)
        self.model = EncoderRNN(
            vocab_size=len(embeddings),
            embedding_dim=embedding_dim,
            rnn_hidden_dim=encoder_hidden_dim,
            dense_hidden_dim=dense_hidden_dim,
            num_classes=len(labels),
            dropout=dropout,
            pretrained_emb=embeddings,
        )

        # init labels, not actually necessary. delete later
        # self.label_to_tensor = {}
        # for i, label in enumerate(labels):
        #     # label_one_hot = torch.LongTensor([0 if k!=i else 1 for k in range(len(labels))])
        #     # self.label_to_tensor[label] = autograd.Variable(label_one_hot.view(1, -1))
        #     self.label_to_tensor[label] = autograd.Variable(torch.LongTensor([i]))

        # init loss function and optimizer
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-3)

    def get_word_idx(self, word):
        try:
            return self.word_to_idx[word]
        except KeyError:
            return self.word_to_idx[self.UNK]

    # noinspection PyArgumentList
    def prepare_sequence(self, word_seq, cuda=True):
        idxs = [self.get_word_idx(word) for word in word_seq]
        if cuda:
            tensor = torch.cuda.LongTensor(idxs)
        else:
            tensor = torch.LongTensor(idxs)
        return autograd.Variable(tensor)

    @staticmethod
    def prepare_labels(labels, cuda=True, evalu=False):
        outputs = [torch.LongTensor([label]) for label in labels]
        if cuda:
            outputs = [output.cuda() for output in outputs]
        return map(lambda out: autograd.Variable(out, volatile=evalu), outputs)

    def predict(self, x, cuda=True):
        super(SentiBiRNN, self).predict(x)
        # self.model.eval()
        preds = []
        for sample in x:
            seq = self.prepare_sequence(sample, cuda=cuda)
            preds.append(self.model.forward(seq))
        return preds

    def predict_loss_and_acc(self, x, gold, cuda=True):
        preds = self.predict(x[:10])
        gold_labels = self.prepare_labels(gold[:10], cuda=cuda, evalu=True)

        losses = []
        for pred, label in zip(preds, gold_labels):
            losses.append(self.loss_function(pred, label).data[0])

        return np.mean(losses)

    def train(self, x, y, reset=True, cuda=True):
        """
        This function will train our RNN model.
        TODO: allow for mini-batch input! Now it only takes one sample at a time!
        """
        super(SentiBiRNN, self).train(x, y)
        # Step 1, clear out the history of the gradient from previous sample,
        # should always be doing this for each sample/MB.
        if reset:
            self.model.zero_grad()
            self.model.init_hiddens()

        # Step 2, prepare input variables.
        # seqs = [self.prepare_sequence(wseq) for wseq in x] # may need this for mbatch functionality
        seq = self.prepare_sequence(x, cuda)
        target = self.prepare_labels([y], cuda)[0]

        # Step 3, feed forward, then backprop!
        outputs = self.model.forward(seq)
        loss = self.loss_function(outputs, target)
        loss.backward()
        self.optimizer.step()


# This is our RNN!
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_hidden_dim, dense_hidden_dim,
                 num_classes, dropout=0.25, pretrained_emb=None):
        super(EncoderRNN, self).__init__()

        # initialize word embedding table
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_emb is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_emb))

        # initialize hidden state h0, diff for forward and backward
        self.hidden_dim = rnn_hidden_dim
        self.hidden_f, self.hidden_b = None, None
        self.init_hiddens()

        # initialize RNNs - two separate ones learnt for forward and backward
        self.rnn_f = nn.GRUCell(embedding_dim, rnn_hidden_dim) # forward rnn
        self.rnn_b = nn.GRUCell(embedding_dim, rnn_hidden_dim) # backward rnn

        # build the sequence
        self.drop = nn.Dropout(p=dropout)
        dense_input_size = 2 * rnn_hidden_dim  # because 2 concatenated hidden states are fed

        # figure out to do minibatches
        self.dense_to_pred_layers = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(dense_input_size, dense_hidden_dim)),
            ('relu1', nn.ReLU()), # nonlinearity, probs relu is best
            ('drop-global1', self.drop), # dropout from main dense layer
            ('dense2', nn.Linear(dense_hidden_dim, num_classes,)),
            ('softmax', nn.Softmax()),
        ]))

    def init_hiddens(self):
        self.hidden_f = self._build_init_hidden()
        self.hidden_b = self._build_init_hidden()

    def _build_init_hidden(self):
        return autograd.Variable(torch.zeros(1, self.hidden_dim).cuda())

    def forward(self, sequence):
        """
        :param sequence: tensor of a sequence of word-to-idxs
        :return: returns final output prediction of model
        """
        embedding_seq = self.word_embeddings(sequence)
        hf = self.hidden_f
        hb = self.hidden_b
        for i in range(len(sequence)):
            hf = self.rnn_f(embedding_seq[i].view(1, -1), hf) # start at beginning of seq
            hb = self.rnn_b(embedding_seq[-1-i].view(1, -1), hb) # "" end of seq

        # TODO: ensure this is concatenating on correct dim!!
        return self.dense_to_pred_layers(torch.cat([hf, hb], 1)) #concat on first axis
