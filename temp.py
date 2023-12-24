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


#read the matrix from csv file 'test_data/num21.csv'
weight_matrix = np.genfromtxt('m.csv', delimiter=',')

#construct the network
network = nx.from_numpy_matrix(weight_matrix)

# network = nx.generators.fast_gnp_random_graph(82,.5)
#construct the DGL network
# scr = np.array([i[0] for i in network.edges])
# des = np.array([i[1] for i in network.edges])
G = dgl.from_networkx(network)

print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())

#add the weights to edges of the DGL constructed graph
# weights = torch.Tensor([i[2]['weight'] for i in list(network.edges(data = True))])
edges = G.edges()
src = edges[0]
dis = edges[1]

G.edata['w'] = torch.randn(len(src))

print("setting up the weights")
for i,j in tqdm(zip(src,dis)):
    feature = network.get_edge_data(int(i),int(j))
    G.edata['w'][G.edge_id(int(i),int(j))] = feature['weight']

#add arbitrary features to nodes
embed = nn.Embedding(82, 5)  # 82 nodes with embedding dim equal to 5
G.ndata['feat'] = embed.weight

#make the class
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

#perform the K-fold
k = len(network) #10
kf = KFold(n_splits=k,shuffle=True)

inputs = embed.weight
X = torch.tensor(range(0,len(network)))
y = torch.LongTensor(np.concatenate([np.zeros(38),np.ones(82-38)]))

accuracy = []
# fscore = []
counter = 0
nets = []
netcounter = 0
missclassifieds = []
for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index,"\n")
    labeled_nodes, X_test = X[train_index], X[test_index]
    labels, y_test = y[train_index], y[test_index]

    #make the neural network
    nets.append(GCN(5, 5, 2))
    net = nets[netcounter]
    netcounter+=1

    optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
    for epoch in range(150):
        logits = net(G, inputs)
        logp = F.log_softmax(logits, 1)
        # we only compute loss for labeled nodes
        loss = F.nll_loss(logp[labeled_nodes], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    output = logits.detach().argmax(axis=1)
    temp_accuracy = accuracy_score(y_test, output[X_test])
    # temp_fscore = f1_score(y_test, output[X_test])
    accuracy.append(temp_accuracy)
    # fscore.append(temp_fscore)
    if temp_accuracy<1:
        for i,j in enumerate(output[X_test]):
            if j != y_test[i]:
                counter+=1
                missclassifieds.append(X_test[i])


    print("in {} we have Accuracy:{}".format(X_test,temp_accuracy))

print("overall accuracy:{}".format(np.array(accuracy).mean()))
print("number of missclassification:{}".format(counter))
print("these were misclassified : ",missclassifieds)