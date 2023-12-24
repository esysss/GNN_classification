import dgl
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from sklearn.model_selection import KFold
from tqdm import tqdm
import itertools
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle

#make the class

def mother(G, inputs,labeled_nodes,labels,embed):
    class GCN(nn.Module):
        # The first layer transforms input features of size of 5 to a hidden size of 5.
        # The second layer transforms the hidden layer and produces output features of
        # size 2, corresponding to the two groups of outputs.
        def __init__(self, in_feats, hidden_size, num_classes):
            super(GCN, self).__init__()
            self.conv1 = GraphConv(in_feats, hidden_size)
            self.conv2 = GraphConv(hidden_size, num_classes)

        def forward(self, g, inputs):
            h = self.conv1(g, inputs)
            h = torch.relu(h)
            h = self.conv2(g, h)
            return h



    net = GCN(20, 20, 10)

    optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)

    for epoch in range(300):
        logits = net(G, inputs)
        logp = F.log_softmax(logits, 1)
        # we only compute loss for labeled nodes
        loss = F.nll_loss(logp[labeled_nodes], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    output = logits.detach().argmax(axis=1)

    return output