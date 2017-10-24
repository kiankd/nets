import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from nets import AbstractModel
from collections import OrderedDict
from torch.nn.utils import rnn
from torch.autograd import Variable

np.random.seed(1)
def random_word_emb(dim):
    return 0.2 * np.random.uniform(-1.0, 1.0, dim)

# Wrapper class over our abstract model paradigm.
class SentiBiRNN(AbstractModel):
    UNK = '*UNKNOWN*'

    def __init__(self, name, vocabulary, embedding_dim, encoder_hidden_dim, dense_hidden_dim,
                 labels, dropout, batch_size, learning_rate=0.01, hyperparameters=None, emb_dict=None):

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
            batch_size=batch_size,
            pretrained_emb=embeddings,
        )

        # init loss function and optimizer
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-3)

    def get_word_idx(self, word):
        try:
            return self.word_to_idx[word]
        except KeyError:
            return self.word_to_idx[self.UNK]

    def prepare_sequences(self, seq_batch, cuda=True):
        seq_batch = [[self.get_word_idx(word) for word in seq] for seq in seq_batch]
        seq_batch = list(reversed(sorted(seq_batch, key=lambda l: len(l))))
        lengths = list(map(len, seq_batch))
        seq_batch = [Variable(torch.LongTensor(x)) for x in seq_batch]
        if cuda:
            seq_batch = [x.cuda() for x in seq_batch]
        return seq_batch, lengths

        # packed_padded_seq = rnn.pack_padded_sequence(seq_batch, lengths)
        #
        # # seq_batch = Variable(torch.LongTensor(seq_batch))
        #
        # print(packed_padded_seq)
        # # padding is essential for doing batches of sequence data
        # # packed_padded_seq = rnn.pack_padded_sequence(seq_batch, lengths)
        #
        # if cuda:
        #     return packed_padded_seq.cuda()
        # return packed_padded_seq

    @staticmethod
    def prepare_labels(labels, cuda=True, evalu=False):
        outs = Variable(torch.LongTensor(labels), volatile=evalu)
        if cuda:
            return outs.cuda()
        return outs

        # outputs = [torch.LongTensor([label]) for label in labels]
        # if cuda:
        #     outputs = [output.cuda() for output in outputs]
        # return map(lambda out: Variable(out, volatile=evalu), outputs)

    def predict(self, x, cuda=True):
        super(SentiBiRNN, self).predict(x)
        self.model.init_hiddens(len(x)) # init for big batch input
        return self.model(*self.prepare_sequences(x, cuda=cuda))

    def predict_loss_and_acc(self, x, gold, cuda=True):
        preds = self.predict(x)
        gold_labels = self.prepare_labels(gold, cuda=cuda, evalu=True)
        return self.loss_function(preds, gold_labels)

    def train(self, x, y, batch_size=-1, reset=True, cuda=True):
        """
        This function will train our RNN model.
        TODO: allow for mini-batch input! Now it only takes one sample at a time!
        """
        super(SentiBiRNN, self).train(x, y)
        # Step 1, clear out the history of the gradient from previous sample,
        # should always be doing this for each sample/MB.
        if reset:
            self.model.zero_grad()
            self.model.init_hiddens(batch_size)

        # Step 2, prepare input variables.
        # seqs = [self.prepare_sequence(wseq) for wseq in x] # may need this for mbatch functionality
        seqs, lens = self.prepare_sequences(x, cuda)
        targets = self.prepare_labels(y, cuda)

        # Step 3, feed forward, then backprop!
        outputs = self.model.forward(seqs, lens)
        loss = self.loss_function(outputs, targets)
        loss.backward()
        self.optimizer.step()


# This is our RNN!
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_hidden_dim, dense_hidden_dim,
                 num_classes, batch_size, dropout=0.25, pretrained_emb=None):
        super(EncoderRNN, self).__init__()
        self.bsz = batch_size
        self.embsz = embedding_dim

        # initialize word embedding table
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_emb is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_emb))

        # initialize hidden state h0, diff for forward and backward
        self.hidden_dim = rnn_hidden_dim
        self.hidden_f, self.hidden_b = None, None
        self.init_hiddens(size=batch_size)

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

    def init_hiddens(self, size):
        self.hidden_f = self._build_init_hidden(size)
        self.hidden_b = self._build_init_hidden(size)

    def _build_init_hidden(self, size):
        return Variable(torch.zeros(size, self.hidden_dim).cuda())

    def forward(self, sorted_seqs, sorted_lens):
        """
        :param sorted_seqs: tensor of a sequence of word-to-idxs
        :param sorted_lens: sorted list of lengths
        :return: returns final output prediction of model
        """
        max_len = sorted_lens[0]
        emb_seq_list = [self.word_embeddings(seq) for seq in sorted_seqs]

        # fucking bullshit that pytorch doesn't come with this
        def pad(tensor, length):
            inc = max_len - length # how much padding we need
            d2_input = tensor.view(1, 1, -1, self.embsz)
            return F.pad(d2_input, (0, 0, inc, 0)).view(max_len, self.embsz) # pad and bring back

        # map the padding to the input
        padded_emb_seqs = [pad(seq, crtlen) for seq, crtlen in zip(emb_seq_list, sorted_lens)]
        padded_embs = torch.stack(padded_emb_seqs)

        # prop through the bi-rnn
        hf = self.hidden_f
        hb = self.hidden_b
        for i in range(max_len):
            hf = self.rnn_f(padded_embs[:,i,:], hf) # start at beginning of seq
            hb = self.rnn_b(padded_embs[:,-1-i,:], hb) # "" end of seq
        rep = torch.cat([hf, hb], 1)
        return self.dense_to_pred_layers(rep)
