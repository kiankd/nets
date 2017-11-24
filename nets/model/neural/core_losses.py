import torch
import numpy as np
from torch import nn

class CORECentroidLoss(nn.Module):

    def __init__(self, lam1, lam2, lam3):
        super(CORECentroidLoss, self).__init__()
        self.lam1 = lam1 # sample-to-centroid attraction
        self.lam2 = lam2 # sample-to-centroid repulsion
        self.lam3 = lam3 # centroid-to-centroid repulsion

    def forward(self, E, h_representations, labels):
        """
        :param E: centroid tensor of size k x h
        :param h_representations: Tensor of size n x h
        :param labels: vector of labels
        :return: loss
        """
        k = E.size()[0]
        n = h_representations.size()[0]

        # cEntroid matrix E, k x h
        enorm = torch.norm(E, p=2, dim=1)

        # dot products between hidden vecs and centroids, n x k
        D = torch.mm(h_representations, E.transpose(0, 1))

        # outer product of two norm vectors, Nij = ||hi|| * ||ej||, n x k
        N = torch.mm(torch.norm(h_representations, p=2, dim=1).view(-1, 1),
                     enorm.view(1, -1))

        # compute cosine similarities (element wise), n x k
        S = torch.mul(D, 1 / N) # turn N 1 over itself -- 1 / (||hi|| * ||ej||)

        # cosine distances, n x k
        C = 0.5 * (-S)#(torch.ones(n, k) - S)

        # mask matrix for the attractive and repulsive, n x k
        T = torch.ones(n, k)
        T = T * (-self.lam2 / ((n - 1) * k)) # there are n-1 * k repulsive signals, neg to maximize
        T[torch.arange(0, n).long(), torch.LongTensor(labels)] = self.lam1 / n # there are n attractive signals

        # final loss
        loss = torch.sum(torch.mul(T, C))

        # centroid-to-centroid loss
        if self.lam3:
            ED = torch.mm(E, E.transpose(0, 1)) # k by k dot prods
            EN = torch.mm(enorm.view(-1, 1), enorm.view(1, -1)) # norms
            ES = torch.mul(ED, EN) # cosine sims
            EC = 0.5 * (torch.ones(k, k) - ES) # cosine dists

            # set the indices to consider
            ET = torch.ones(k, k)
            ET = ET.triu()
            ET[np.arange(k), np.arange(k)] = 0 # zero out dists btwn vec and itself
            normalizer = self.lam3 / torch.sum(ET)

            # update the loss to include this
            # negative because we want to maximize dist between centroids
            loss += -normalizer * torch.sum((torch.mul(ET, EC)))

        return loss
