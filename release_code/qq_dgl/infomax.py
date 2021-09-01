import os.path as osp
import math 
import pdb 
import torch

import torch.nn as nn
import torch.nn.functional as F
import json
from torch_geometric.utils import to_dense_batch, to_dense_adj


def get_positive_expectation(p_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        try:
            Eq = F.softplus(-q_samples) + q_samples - log_2
        except RuntimeError as e:
            pdb.set_trace()
            tmp = 1
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq


def local_global_drug_loss_(l_enc, g_enc, adj_tensor, num_drugs, measure, device):
#     if args.cuda:
#         pos_mask = adj_tensor.cuda() + torch.eye(num_drugs).cuda()
#         neg_mask = torch.ones((num_drugs, num_drugs)).cuda() - pos_mask
#     else:
#         pos_mask = adj_tensor + torch.eye(num_drugs)
#         neg_mask = torch.ones((num_drugs, num_drugs)) - pos_mask
    num_edges_w = 12236
    pos_mask = torch.eye(num_drugs) + adj_tensor
    neg_mask = torch.ones((num_drugs, num_drugs)) - pos_mask
#     pos_mask.to(device)
#     neg_mask.to(device)
    
    res = torch.mm(l_enc, g_enc.t())
    #res = res.to(device)
    
    num_edges = num_edges_w + num_drugs
    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_edges
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_drugs ** 2 - 2 * num_edges)
    return (E_neg - E_pos)


# 256
class FF_local(torch.nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        FF_hidden1, FF_hidden2, FF_output = 32, 32, 32
        self.block = nn.Sequential(
            nn.Linear(input_dim, FF_hidden1),
            nn.ReLU(),
            nn.Linear(FF_hidden1, FF_hidden2),
            nn.ReLU(),
            nn.Linear(FF_hidden2, FF_output),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, FF_output)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


# 128
class FF_global(torch.nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        FF_hidden1, FF_hidden2, FF_output = 32, 32, 32
        self.block = nn.Sequential(
            nn.Linear(input_dim, FF_hidden1),
            nn.ReLU(),
            nn.Linear(FF_hidden1, FF_hidden2),
            nn.ReLU(),
            nn.Linear(FF_hidden2, FF_output),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, FF_output)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class shortcut(nn.Module):

    def __init__(self, input_dim_1, input_dim_2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim_1, input_dim_2),
            nn.ReLU(),
        )
        self.linear_shortcut = nn.Linear(input_dim_2, input_dim_2)

    def forward(self, x1, x2):
        return self.block(x1) + self.linear_shortcut(x2)


class GcnInfoMax(torch.nn.Module):

    def __init__(self, gamma, prior, features_dim, embedding_dim, device):

        super(GcnInfoMax, self).__init__()
        self.gamma = gamma
        self.prior = prior
        self.features_dim = features_dim
        self.embedding_dim = embedding_dim
        self.device = device 
        self.local_d = FF_local(self.features_dim)
        self.global_d = FF_global(self.embedding_dim)

    def forward(self, global_embeddings, loc_embeddings, adj_tensor, num_drugs):
    
        g_enc = self.global_d(global_embeddings)
        l_enc = self.local_d(loc_embeddings)
        measure = 'JSD' # ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
        local_global_loss = local_global_drug_loss_(l_enc, g_enc, adj_tensor, num_drugs, measure, self.device)
        eps = 1e-5
        PRIOR = 0

        return local_global_loss + PRIOR
    
#         if self.prior:
#             prior = torch.rand_like(embeddings)
#             term_a = torch.log(self.prior_d(prior) + eps).mean()
#             term_b = torch.log(1.0 - self.prior_d(embeddings) + eps).mean()
#             PRIOR = - (term_a + term_b) * self.gamma
#         else:
#             PRIOR = 0
