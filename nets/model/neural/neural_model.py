import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from nets import AbstractModel, util
from nets.model.neural import clip_gradient
from collections import OrderedDict, Counter
from itertools import izip
from torch.autograd import Variable

#######################
## Parameter statics ##
#######################
LOSS_FUN = 'loss'
LEARNING_RATE = 'lr'
CLIP_NORM = 'cn'
WEIGHT_DECAY = 'decay'

class NeuralModel(AbstractModel):
    def __init__(self, name, torch_module, params):
        super(NeuralModel, self).__init__(name, params)
        self.model = torch_module
        self.params = params
        self.optimizer = optim.Adam(self.parameters(),
                                    lr=self.params[LEARNING_RATE],
                                    weight_decay=self.params[WEIGHT_DECAY])

    @staticmethod
    def prepare_samples(sample_batch, cuda=True):
        samps = [Variable(torch.LongTensor(x)) for x in sample_batch]
        if cuda:
            samps = [x.cuda() for x in seq_batch]
        return samps

    @staticmethod
    def prepare_labels(labels, cuda=True, evalu=False):
        outs = Variable(torch.LongTensor(labels), volatile=evalu)
        if cuda:
            return outs.cuda()
        return outs

    @staticmethod
    def get_accuracy(output, golds):
        return [(name, f(golds, output)) for name, f in util.iter_accs()]

    def parameters(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def predict(self, x):
        super(NeuralModel, self).predict(x)
        return self.model(x)

    def predict_with_stats(self, x, gold, cuda=True):
        preds = self.predict(x)
        golds = self.prepare_labels(gold, cuda=cuda, evalu=True)
        loss = self.params[LOSS_FUN](preds, golds)

        preds = preds.data.type(torch.DoubleTensor).numpy()
        golds = golds.data.type(torch.DoubleTensor).numpy()

        output = np.argmax(preds, axis=1)

        mean_pred_label_conf = np.mean(np.max(preds, axis=0)) # e.g., avg. conf for l=0 is 0.6, l=1 is 0.4
        return loss, \
               self.get_accuracy(output, golds), \
               dict(Counter(list(output))), \
               mean_pred_label_conf

    def train(self, x, y, cuda=True):
        """
        This function will train our RNN model over a minibatch.
        """
        super(NeuralModel, self).train(x, y)
        # Step 1, clear out the history of the gradient from previous sample,
        # should always be doing this for each sample/MB.
        self.model.zero_grad()

        # Step 2, prepare input variables.
        samples = self.prepare_samples(x, cuda)
        targets = self.prepare_labels(y, cuda)

        # Step 3, feed forward, then backprop!
        outputs = self.model(samples)
        loss = self.params[LOSS_FUN](outputs, targets)
        loss.backward()

        # Step 4 - optional grad clips
        if self.params[CLIP_NORM] > 0:
            coeff = clip_gradient(self.parameters(), self.clip_norm)
            for p in self.parameters():
                p.grad.mul_(coeff)

        self.optimizer.step()

    def get_w_norm(self):
        norm = 0
        for param in self.parameters():
            norm += torch.norm(param)
        return norm.data[0]

    def get_grad_norm(self):
        norm = 0
        for param in self.parameters():
            try:
                norm += param.grad.data.norm()
            except AttributeError:
                norm += 0
        if norm == 0:
            return 'No Grad...'
        return norm
