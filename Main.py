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
import model
import scipy.io

mat = scipy.io.loadmat('test_data/Fc.mat')


network = nx.from_numpy_matrix(mat['FC'])
# network = nx.generators.fast_gnp_random_graph(82,.5)
#construct the DGL network
# scr = np.array([i[0] for i in network.edges])
# des = np.array([i[1] for i in network.edges])
G = dgl.from_networkx(network)
G = dgl.add_self_loop(G)

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



#perform the K-fold
k = len(network) #10
kf = KFold(n_splits=k,shuffle=True)

X = torch.tensor(range(0,len(network)))
labels = np.concatenate([np.zeros(38),np.ones(82-38)])
y = torch.LongTensor(labels)

accuracy = []
# fscore = []
counter = 0
missclassifieds = []
for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index,"\n")
    labeled_nodes, X_test = X[train_index], X[test_index]
    labels, y_test = y[train_index], y[test_index]

    G.edata['w'] = torch.randn(len(src))

    print("setting up the weights")
    for i, j in zip(src, dis):
        feature = network.get_edge_data(int(i), int(j))
        G.edata['w'][G.edge_id(int(i), int(j))] = feature['weight']
    # add arbitrary features to nodes
    embed = nn.Embedding(len(network), 20)  # nodes with embedding features
    G.ndata['feat'] = embed.weight
    inputs = embed.weight

    #make the neural network
    output = model.mother(G, inputs,labeled_nodes,labels,embed)
    temp_accuracy = accuracy_score(y_test, output[X_test])
    # temp_fscore = f1_score(y_test, output[X_test])
    accuracy.append(temp_accuracy)
    # fscore.append(temp_fscore)
    if temp_accuracy<1:
        for i,j in enumerate(output[X_test]):
            if j != y_test[i]:
                counter+=1
                missclassifieds.append(X_test[i])

    print("we have Accuracy:{}".format(temp_accuracy))

print("overall accuracy:{}".format(np.array(accuracy).mean()))
print("number of missclassification:{}".format(counter))
# print("these were misclassified : ",missclassifieds)