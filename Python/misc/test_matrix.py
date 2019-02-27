from __future__ import division
import numpy as np
import networkx as nx

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


n_genes=50
covs = [0.1,0.2, 0.3,0.4]
for cov in covs:
    values = []
    for i in list(range(10000)):
        ntwk = nx.barabasi_albert_graph(n_genes, 2)
        ntwk_np = nx.to_numpy_matrix(ntwk)
        C = ntwk_np * cov
        np.fill_diagonal(C, 1)
        values.append(is_pos_def(C))
    print(cov, sum(values)/len(values))
