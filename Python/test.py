import parevol_tools as pt
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools, random

c = np.array([[1,2],[3,0],[0,2]])

c_new = pt.get_random_matrix(c)

#print(c)
#print(np.mean(c, axis=0))
#print(c - np.mean(c, axis=0))



ntwk = nx.barabasi_albert_graph(100, 2)
ntwk_np = nx.to_numpy_matrix(ntwk)
k_ = ntwk_np.sum(axis=0)
k_ = np.sort(k_)
k_ = np.asarray(k_.tolist()[0])

rates = np.random.gamma(1, scale=1, size=100)
rates = np.sort(rates)




#pairs = list(itertools.combinations(range(4), 2))
#random.shuffle(pairs)
#to_sample = 2
#sample_pairs = pairs[:to_sample]
#print(sample_pairs)
#for sample_pair in sample_pairs:
#    #i[sample_pair[0]], i[sample_pair[1]] = i[a], i[b]



#random values covariance matrix
cov=0.2
C = pt.get_ba_cov_matrix(5, cov)
diag_C = np.tril(C, k =-1)
print(C)
i,j = np.nonzero(diag_C)
# remove redundant pairs

prop=0.2
ix = np.random.choice(len(i), int(np.floor((1-prop) * len(i))), replace=False)
#print(ix)
#for ixx in ix:
print(i[ix], j[ix])
print(C[i[ix], j[ix]])

C[np.concatenate((i[ix],j[ix]), axis=None), np.concatenate((j[ix],i[ix]), axis=None)] = -1*cov
print(C)
print(np.all(np.linalg.eigvals(C) > 0))
