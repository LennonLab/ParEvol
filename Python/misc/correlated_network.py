from __future__ import division
import math, random
import numpy as np
import networkx as nx
import parevol_tools as pt
import matplotlib.pyplot as plt




def get_correlated_rndm_ntwrk_(nodes=100, m=2, rho=0.3, count_threshold = 10000):
    #  Xalvi-Brunet and Sokolov
    # generate maximally correlated networks with a predefined degree sequence
    if rho > 0:
        assortative = True
    elif rho < 0:
        assortative = False
    else:
        print("rho can't be zero")

    def get_two_edges(graph_array):
        d = nx.to_dict_of_dicts(nx.from_numpy_matrix(graph_array), edge_data=1)
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

    iter_rh0 = 0
    iter_graph = None
    while (assortative == True and iter_rh0 < rho) or (assortative == False and iter_rh0 > rho):
        count = 0
        current_rho = 0
        accepted_counts = 0
        graph = nx.barabasi_albert_graph(nodes, m)
        graph_np = nx.to_numpy_matrix(graph)
        while ((assortative == True and current_rho < rho) or (assortative == False and current_rho > rho)) and ((count-accepted_counts) < count_threshold):
        #while (abs(current_rho) < abs(rho)) and ((count-accepted_counts) < count_threshold) : #<r (rejected_counts < count_threshold):
            count += 1
            edges = get_two_edges(graph_np)
            graph_np_sums = np.sum(graph_np, axis=1)
            # check whether new edges already exist
            if graph_np[edges[0][0],edges[3][0]] == 1 or \
                graph_np[edges[3][0],edges[0][0]] == 1 or \
                graph_np[edges[2][0],edges[1][0]] == 1 or \
                graph_np[edges[1][0],edges[2][0]] == 1:
                continue

            disc = (edges[0][1] - edges[2][1]) * \
                    (edges[3][1] - edges[1][1])
            #if (rho > 0 and disc > 0) or (rho < 0 and disc < 0):
            if (assortative == True and disc > 0) or (assortative == False and disc < 0):
                graph_np[edges[0][0],edges[1][0]] = 0
                graph_np[edges[1][0],edges[0][0]] = 0
                graph_np[edges[2][0],edges[3][0]] = 0
                graph_np[edges[3][0],edges[2][0]] = 0

                graph_np[edges[0][0],edges[3][0]] = 1
                graph_np[edges[3][0],edges[0][0]] = 1
                graph_np[edges[2][0],edges[1][0]] = 1
                graph_np[edges[1][0],edges[2][0]] = 1

                accepted_counts += 1
                current_rho = nx.degree_assortativity_coefficient(nx.from_numpy_matrix(graph_np))
                #print(current_rho, count, accepted_counts)

        iter_rh0 = nx.degree_assortativity_coefficient(nx.from_numpy_matrix(graph_np))
        iter_graph = graph_np

    return iter_graph



    #graph_np = graph_np.astype(int)

    #return graph_np.astype(int)





#  Xalvi-Brunet and Sokolov
# generate maximally correlated networks with a predefined degree sequence
def get_correlated_rndm_ntwrk(nodes=100, m=2, rho=0.3, count_threshold = 10000):
    if rho > 0:
        assortative = True
    elif rho < 0:
        assortative = False
    else:
        print("rho can't be zero")

    def get_two_edges(graph_array):
        d = nx.to_dict_of_dicts(nx.from_numpy_matrix(graph_array), edge_data=1)
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

    #def run_sim():
    #    count = 0
    #    current_rho = 0
    #    accepted_counts = 0

    while True:
        count = 0
        current_rho = 0
        accepted_counts = 0
        graph = nx.barabasi_albert_graph(nodes, m)
        graph_np = nx.to_numpy_matrix(graph)
        while ((assortative == True and current_rho < rho) or (assortative == False and current_rho > rho)) and ((count-accepted_counts) < count_threshold):
        #while (abs(current_rho) < abs(rho)) and ((count-accepted_counts) < count_threshold) : #<r (rejected_counts < count_threshold):
            count += 1
            edges = get_two_edges(graph_np)
            graph_np_sums = np.sum(graph_np, axis=1)
            # check whether new edges already exist
            if graph_np[edges[0][0],edges[3][0]] == 1 or \
                graph_np[edges[3][0],edges[0][0]] == 1 or \
                graph_np[edges[2][0],edges[1][0]] == 1 or \
                graph_np[edges[1][0],edges[2][0]] == 1:
                continue

            disc = (edges[0][1] - edges[2][1]) * \
                    (edges[3][1] - edges[1][1])
            #if (rho > 0 and disc > 0) or (rho < 0 and disc < 0):
            if (assortative == True and disc > 0) or (assortative == False and disc < 0):
                graph_np[edges[0][0],edges[1][0]] = 0
                graph_np[edges[1][0],edges[0][0]] = 0
                graph_np[edges[2][0],edges[3][0]] = 0
                graph_np[edges[3][0],edges[2][0]] = 0

                graph_np[edges[0][0],edges[3][0]] = 1
                graph_np[edges[3][0],edges[0][0]] = 1
                graph_np[edges[2][0],edges[1][0]] = 1
                graph_np[edges[1][0],edges[2][0]] = 1

                accepted_counts += 1
                current_rho = nx.degree_assortativity_coefficient(nx.from_numpy_matrix(graph_np))

                print(current_rho, count, accepted_counts)

        return graph_np



    #graph_np = graph_np.astype(int)
    #print(current_rho)

    #if assortative == True:
    #    txt_name = 'assoc_network_eq'
    #else:
    #    txt_name = 'disassoc_network_eq'
    #return graph_np.astype(int)
    #np.savetxt(pt.get_path() + '/data/'+txt_name+'.txt', graph_np.astype(int), delimiter="\t")



def get_correlated_rndm_ntwrk_original(nodes=10, m=2, rho=0.5):
    assort_ = []
    graph = nx.barabasi_albert_graph(nodes, m)
    graph_np = nx.to_numpy_matrix(graph)
    #np.savetxt(pt.get_path() + '/data/disassoc_network_n0.txt', graph_np.astype(int), delimiter="\t")
    #iter = 100
    count = 0
    current_rho = 0
    #while count < iter:
    rejected_counts = 0
    while abs(current_rho) < abs(rho):
        def get_two_edges(graph_array):
            #d = nx.to_dict_of_dicts(graph_array, edge_data=1)
            d = nx.to_dict_of_dicts(nx.from_numpy_matrix(graph_array), edge_data=1)
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
        graph_np_sums = np.sum(graph_np, axis=1)
        #if edges == edges_copy:
        #    continue
        # check whether new edges already exist
        if graph_np[edges[0][0],edges[3][0]] == 1 or \
            graph_np[edges[3][0],edges[0][0]] == 1 or \
            graph_np[edges[2][0],edges[1][0]] == 1 or \
            graph_np[edges[1][0],edges[2][0]] == 1:
            continue

        disc = (edges[0][1] - edges[2][1]) * \
                (edges[3][1] - edges[1][1])
        if (assortative == True and disc > 0) or (assortative == False and disc < 0):
            graph_np[edges[0][0],edges[1][0]] = 0
            graph_np[edges[1][0],edges[0][0]] = 0
            graph_np[edges[2][0],edges[3][0]] = 0
            graph_np[edges[3][0],edges[2][0]] = 0

            graph_np[edges[0][0],edges[3][0]] = 1
            graph_np[edges[3][0],edges[0][0]] = 1
            graph_np[edges[2][0],edges[1][0]] = 1
            graph_np[edges[1][0],edges[2][0]] = 1

            assort_.append(nx.degree_assortativity_coefficient(nx.from_numpy_matrix(graph_np)))
            count += 1
            current_rho = nx.degree_assortativity_coefficient(nx.from_numpy_matrix(graph_np))

            print(current_rho, rejected_counts)

        else:
            rejected_counts += 1
            #print(count, disc, nx.degree_assortativity_coefficient(nx.from_numpy_matrix(graph_np)))

    graph_np.astype(int)
    if assortative == True:
        txt_name = 'assoc_network_eq'
    else:
        txt_name = 'disassoc_network_eq'
    np.savetxt(pt.get_path() + '/data/'+txt_name+'.txt', graph_np.astype(int), delimiter="\t")




def modular_ntwrk():
    G_025 = nx.algorithms.community.LFR_benchmark_graph(n=250, tau1 = 3, tau2 = 1.5, mu = 0.25, average_degree=5, min_community=20, seed=10)
    np.savetxt(pt.get_path() + '/data/modular_ntwrk_mu_025.txt', nx.to_numpy_matrix(G_025), delimiter="\t")

    G_015 = nx.algorithms.community.LFR_benchmark_graph(n=250, tau1 = 3, tau2 = 1.5, mu = 0.15, average_degree=5, min_community=20, seed=10)
    np.savetxt(pt.get_path() + '/data/modular_ntwrk_mu_015.txt', nx.to_numpy_matrix(G_015), delimiter="\t")

    G_010 = nx.algorithms.community.LFR_benchmark_graph(n=250, tau1 = 3, tau2 = 1.5, mu = 0.01, average_degree=5, min_community=20, seed=10)
    np.savetxt(pt.get_path() + '/data/modular_ntwrk_mu_010.txt', nx.to_numpy_matrix(G_010), delimiter="\t")
    communities = {frozenset(G_010.nodes[v]['community']) for v in G_010}
    #print(communities)

    #nx.draw(G_025)
    #plt.savefig(pt.get_path() + '/figs/modular_ntwrk_025.png', format="PNG")
    #nx.draw(G_015)
    #plt.savefig(pt.get_path() + '/figs/modular_ntwrk_015.png', format="PNG")
    #nx.draw(G_010)
    #plt.savefig(pt.get_path() + '/figs/modular_ntwrk_010.png', format="PNG")

#modular_ntwrk()

#get_correlated_rndm_ntwrk(assortative = True)
#get_correlated_rndm_ntwrk(assortative = False, rho = -0.7)

#ntwrk = np.loadtxt(pt.get_path() + '/data/modular_ntwrk_mu_010.txt', delimiter='\t')#, dtype='int')
#cov=0.05
#var=1
#ntwrk = ntwrk * cov
#np.fill_diagonal(ntwrk, var)
#print(np.all(np.linalg.eigvals(ntwrk) > 0))

get_correlated_rndm_ntwrk_()
