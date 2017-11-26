import torch
import numpy as np
from torch import nn
from torch.autograd import Variable

class CORECentroidLoss(nn.Module):

    def __init__(self, lam1, lam2, lam3):
        super(CORECentroidLoss, self).__init__()
        self.lam1 = lam1 # sample-to-centroid attraction
        self.lam2 = lam2 # sample-to-centroid repulsion
        self.lam3 = lam3 # centroid-to-centroid repulsion

    def forward(self, E, h_representations, labels, cuda):
        """
        :param E: centroid tensor variable of size k x h
        :param h_representations: Tensor variable of size n x h
        :param labels: vector of labels
        :param cuda: use gpu
        :return: loss
        """
        k = E.size(0)
        n = h_representations.size(0)

        # cEntroid matrix E, k x h
        enorm = torch.norm(E, p=2, dim=1)

        loss = 0

        if self.lam1 or self.lam2:
            # dot products between hidden vecs and centroids, n x k
            D = torch.mm(h_representations, E.transpose(0, 1))

            # outer product of two norm vectors, Nij = ||hi|| * ||ej||, n x k
            N = torch.mm(torch.norm(h_representations, p=2, dim=1).view(-1, 1),
                         enorm.view(1, -1))

            # compute cosine similarities (element wise), n x k
            S = torch.mul(D, 1 / N) # turn N 1 over itself -- 1 / (||hi|| * ||ej||)

            # cosine distances, n x k
            ones = Variable(torch.ones(n, k).cuda())
            C = 0.5 * (ones - S)

            # mask matrix for the attractive and repulsive, n x k
            t = np.ones((n, k))
            t *= (-self.lam2 / ((n - 1) * k)) # TODO: determine that this should be n * (k-1)!
            t[np.arange(n), labels] = self.lam1 / n
            T = Variable(torch.from_numpy(t).float()).cuda()

            # final loss
            loss += torch.sum(torch.mul(T, C))

        # centroid-to-centroid loss
        if self.lam3:
            ED = torch.mm(E, E.transpose(0, 1)) # k by k dot prods
            EN = torch.mm(enorm.view(-1, 1), enorm.view(1, -1)) # norms
            ES = torch.mul(ED, 1 / EN) # cosine sims
            ones = Variable(torch.ones(k, k).cuda())
            EC = 0.5 * (ones - ES) # cosine dists, no need for the -1

            # set the indices to consider
            et = np.triu(np.ones((k, k)))
            np.fill_diagonal(et, 0)
            ET = Variable(torch.from_numpy(et).float()).cuda()
            normalizer = self.lam3 / torch.sum(ET)

            # update the loss to include this
            # negative because we want to maximize dist between centroids
            loss += - (normalizer * torch.sum((torch.mul(ET, EC))))

        return loss
