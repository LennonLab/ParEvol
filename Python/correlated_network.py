from __future__ import division
import math, random, operator
import numpy as np
import networkx as nx
import parevol_tools as pt
import matplotlib.pyplot as plt

#  Xalvi-Brunet and Sokolov
# generate maximally correlated networks with a predefined degree sequence
def get_correlated_rndm_ntwrk(assortative = False):
    assort_ = []
    graph = nx.barabasi_albert_graph(100, 3)
    graph_np = nx.to_numpy_matrix(graph)
    np.savetxt(pt.get_path() + '/data/disassoc_network_n0.txt', graph_np.astype(int), delimiter="\t")
    iter = 100
    count = 0
    while count < iter:
    #for j in range(iter):
        def get_two_edges(graph_array):
            d = nx.to_dict_of_dicts(graph,edge_data=1)
            l0_n0 = random.sample(list(d), 1)[0]
            l0_list = list(d[l0_n0])
            l0_n1 = random.sample(l0_list, 1)[0]

            def get_second_edge(d, l0_n0, l0_n1):
                l1_list = [i for i in list(d) if i not in [l0_n0, l0_n1] ]
                l1 = []
                while len(l1) != 2:
                    l1_n0 = random.sample(list(l1_list), 1)[0]
                    l1_n1_list = d[l1_n0]
                    l1_n1_list = [i for i in l1_n1_list if i not in [l0_n0, l0_n1] ]
                    if len(l1_n1_list) > 0:
                        l1_n1 = random.sample(list(l1_n1_list), 1)[0]
                        l1.extend([l1_n0, l1_n1])
                return l1

            # get two links, make sure all four nodes are unique
            link1 = get_second_edge(d, l0_n0, l0_n1 )
            row_sums = np.asarray(np.sum(graph_array, axis =0))[0]
            node_edge_counts = [(l0_n0, row_sums[l0_n0]), (l0_n1, row_sums[l0_n1]),
                                (link1[0], row_sums[link1[0]]), (link1[1], row_sums[link1[1]])]
            return node_edge_counts

        edges = get_two_edges(graph_np)
        edges_copy = edges.copy()
        edges_copy.sort(key=operator.itemgetter(1))
        #sums = np.sum(graph_np, axis=1)

        if edges == edges_copy:
            continue

        #print(graph_np[edges[0][0],edges[1][0]], graph_np[edges[2][0],edges[3][0]])

        graph_np[edges[0][0],edges[1][0]] = 0
        graph_np[edges[2][0],edges[3][0]] = 0

        if assortative == True:
            graph_np[edges_copy[0][0],edges_copy[1][0]] = 1
            graph_np[edges_copy[2][0],edges_copy[3][0]] = 1

        else:# Disassortative
            graph_np[edges_copy[0][0],edges_copy[3][0]] = 1
            graph_np[edges_copy[1][0],edges_copy[2][0]] = 1

        count += 1

        print(count, nx.degree_assortativity_coefficient(nx.from_numpy_matrix(graph_np)))
        assort_.append(nx.degree_assortativity_coefficient(nx.from_numpy_matrix(graph_np)))


    graph_np.astype(int)
    np.savetxt(pt.get_path() + '/data/disassoc_network_eq.txt', graph_np.astype(int), delimiter="\t")

    #fig = plt.figure()
    #plt.scatter(list(range(iter)), assort_, c='#175ac6', marker = 'o', s = 70, \
    #    edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)#, edgecolors='none')
    #plt.xlabel ('Iteration', fontsize=18)
    #plt.ylabel ('Degree of network (Dis)assortativity, ' + r'$r$', fontsize=14)
    #plt.title('Disassortative model of\nXulvi-Brunet - Sokolov algorithm, ' + r'$p=1$', fontsize=14)
    #fig.tight_layout()
    #fig.savefig(pt.get_path() + '/figs/test_cor_net_d.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    #plt.close()


get_correlated_rndm_ntwrk()

#eq_ = np.loadtxt(pt.get_path() + '/data/disassoc_network_eq.txt', dtype='float', delimiter='\t')
#n0_ = np.loadtxt(pt.get_path() + '/data/disassoc_network_n0.txt', dtype='float', delimiter='\t')

#print(np.sum(eq_, axis=1))
#print(np.sum(n0_, axis=1))
