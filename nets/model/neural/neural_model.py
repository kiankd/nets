import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from nets import AbstractModel, util
from nets.model.neural import clip_gradient
from nets.model.neural.core_losses import CORECentroidLoss
from collections import OrderedDict, Counter
from torch.autograd import Variable

#######################
## Parameter statics ##
#######################
LOSS_FUN = 'loss'
LEARNING_RATE = 'lr'
CLIP_NORM = 'cn'
WEIGHT_DECAY = 'decay'
CORE = 'core'
LAM1 = 'lam1'
LAM2 = 'lam2'
LAM3 = 'lam3'


class NeuralModel(AbstractModel):
    def __init__(self, name, torch_module, params, unique_labels):
        super(NeuralModel, self).__init__(name, params)
        self.model = torch_module
        self.params = params
        self.unique_labels = unique_labels

        self.lr = self.params[LEARNING_RATE]
        self.cce_loss_function = self.params[LOSS_FUN]()

        self.core = self.params[CORE]
        if len(self.core) == 3:
            self.core_loss_function = CORECentroidLoss(self.core[LAM1], self.core[LAM2], self.core[LAM3])
        elif len(self.core) == 2:
            #TODO: implement pairwise CORE.
            pass
        else:
            self.core_loss_function = None

        self.loss_presentations = [
            CORECentroidLoss(1, 0, 0),
            CORECentroidLoss(0, 1, 0),
            CORECentroidLoss(0, 0, 1),
        ]

        self.optimizer = optim.Adam(self.parameters(),
                                    lr=self.params[LEARNING_RATE],
                                    weight_decay=self.params[WEIGHT_DECAY])

    @staticmethod
    def prepare_samples(sample_batch, cuda=True):
        v = Variable(torch.FloatTensor(sample_batch))
        if cuda:
            return v.cuda()
        return v
        # samps = [Variable(torch.FloatTensor(sample)) for sample in sample_batch]
        # if cuda:
        #     samps = [sample.cuda() for sample in samps]
        # return samps

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

    def update_lr(self, change):
        self.lr *= change
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def training_predict(self, x):
        super(NeuralModel, self).predict(x)
        samples = self.prepare_samples(x)
        return self.model(samples)

    def predict(self, x):
        preds, reps = self.training_predict(x)
        preds = preds.data.type(torch.DoubleTensor).numpy()
        return np.argmax(preds, axis=1), reps

    def predict_with_stats(self, x, gold, cuda=True):
        preds, reps = self.training_predict(x)
        targets = self.prepare_labels(gold, cuda=cuda, evalu=True)
        E = self.get_centroids(reps, gold, cuda)

        losses = [self.cce_loss_function(preds, targets)]
        losses += [core_loss(E, reps, gold, cuda) for core_loss in self.loss_presentations]

        preds = preds.data.type(torch.DoubleTensor).numpy()
        targets = targets.data.type(torch.DoubleTensor).numpy()

        output = np.argmax(preds, axis=1)

        mean_pred_label_conf = np.mean(np.max(preds, axis=1)) # e.g., avg. conf for l=0 is 0.6, l=1 is 0.4
        return losses, \
               self.get_accuracy(output, targets), \
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
        outputs, reprs = self.model(samples)

        loss = self.cce_loss_function(outputs, targets)
        if len(self.core) == 3: #h_representations, labels, k
            centroids = self.get_centroids(reprs, y, cuda)
            core_loss = self.core_loss_function(centroids, reprs, y, cuda)
            loss += core_loss

        loss.backward()

        # Step 4 - optional grad clips
        if CLIP_NORM in self.params:
            if self.params[CLIP_NORM] > 0:
                coeff = clip_gradient(self.parameters(), self.clip_norm)
                for p in self.parameters():
                    p.grad.mul_(coeff)

        self.optimizer.step()

    def get_centroids(self, reps, labels, cuda):
        centroids = []
        for label in self.unique_labels:
            idxs = torch.LongTensor(np.where(labels==label)[0])
            if cuda:
                idxs = idxs.cuda()
            center = reps[idxs]
            centroids.append(torch.mean(center, 0).view(-1)) # axis 0
        return torch.stack(centroids)

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
