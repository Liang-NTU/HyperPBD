import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD, Adam, ASGD, RMSprop
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax, softmax
import torch.nn.functional as F
from configparser import ConfigParser
import numpy as np
import math
from torch.nn.parameter import Parameter

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True, activation=True):
        super(HGNN_conv, self).__init__()

        self.weight1 = Parameter(torch.Tensor(in_ft, out_ft))
        self.weight2 = Parameter(torch.Tensor(out_ft, in_ft))
        if bias:
            self.bias1 = Parameter(torch.Tensor(out_ft))
            self.bias2 = Parameter(torch.Tensor(in_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.activation = activation

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        stdv2 = 1. / math.sqrt(self.weight2.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        self.weight2.data.uniform_(-stdv2, stdv2)
        if self.bias1 is not None:
            self.bias1.data.uniform_(-stdv1, stdv1)
            self.bias2.data.uniform_(-stdv2, stdv2)

    def forward(self, x, norm_HH, norm_HG):
        x = x.matmul(self.weight1)
        if self.bias1 is not None:
            x = x + self.bias1

        hyper_emb = torch.sparse.mm(norm_HH, x) # H*D * D*E
        if self.activation:
            hyper_emb = F.relu(hyper_emb)

        z = hyper_emb.matmul(self.weight2)
        if self.bias2 is not None:
            z = z + self.bias2

        x = torch.sparse.mm(norm_HG, z)
        if self.activation:
            x = F.relu(x)
        return x, hyper_emb

class Group_AE(nn.Module):
    def __init__(self, in_ft, out_ft, K, activation=True):
        super(Group_AE, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft

        self.weight1 = Parameter(torch.Tensor(in_ft, out_ft))
        self.weight2 = Parameter(torch.Tensor(out_ft, in_ft))
        self.register_parameter('bias', None)
        
        self.base_g = Parameter(torch.Tensor(out_ft, K))

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        stdv2 = 1. / math.sqrt(self.weight2.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        self.weight2.data.uniform_(-stdv2, stdv2)
        self.base_g.data.uniform_(-stdv1, stdv1)

    def forward(self, x):
        x = x.view(-1, self.in_ft)
        x = x.matmul(self.weight1)
        if self.activation:
            x = F.relu(x)

        lamd = x.matmul(self.base_g)
        lamd = F.softmax(lamd, dim=1)

        encoded = lamd.matmul(self.base_g.T)
        decoded = encoded.matmul(self.weight2)

        return encoded, lamd, decoded

class Score_layer(nn.Module):
    def __init__(self, emb_dim, bias=True, activation=True):
        super(Score_layer, self).__init__()

        self.weight1 = Parameter(torch.Tensor(emb_dim*4, emb_dim))
        self.weight2 = Parameter(torch.Tensor(emb_dim, 1))
        if bias:
            self.bias1 = Parameter(torch.Tensor(emb_dim))
            self.bias2 = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.activation = activation

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        stdv2 = 1. / math.sqrt(self.weight2.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        self.weight2.data.uniform_(-stdv1, stdv1)
        if self.bias1 is not None:
            self.bias1.data.uniform_(-stdv1, stdv1)
            self.bias2.data.uniform_(-stdv1, stdv1)
        
    def forward(self, x):
        x = x.matmul(self.weight1)
        if self.bias1 is not None:
            x = x + self.bias1
        if self.activation:
            x = F.relu(x)

        x = x.matmul(self.weight2)
        if self.bias2 is not None:
            x = x + self.bias2
        if self.activation:
            x = F.relu(x)

        return x