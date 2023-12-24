import networkx as nx
from networkx.algorithms import node_classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

weight_matrix = np.genfromtxt('m.csv', delimiter=',')
print("the shape of matrix is :", weight_matrix.shape)
network = nx.from_numpy_matrix(weight_matrix)

labels = np.concatenate([np.zeros(38), np.ones(82-38)])
print("length of labels is : ", len(labels))

nodes = list(network.nodes())
accuracy = []
counter = 0

k = len(network) #10
kf = KFold(n_splits=k,shuffle=True)
for train_index, test_index in kf.split(nodes):
    network = nx.from_numpy_matrix(weight_matrix)
    for i in train_index:
        network.nodes[i]["label"] = labels[i]

    y_test = labels[test_index]

    output = np.array(node_classification.harmonic_function(network))
    temp_accuracy = accuracy_score(y_test.astype(int), output[test_index].astype(int))
    accuracy.append(temp_accuracy)

    print("in {} we have Accuracy:{}".format(test_index, temp_accuracy))

    if temp_accuracy<1:
        for i,j in enumerate(output[test_index]):
            if j != y_test[i]:
                counter+=1

print("overall accuracy:{}".format(np.array(accuracy).mean()))
print("number of missclassification:{}".format(counter))