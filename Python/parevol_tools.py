from __future__ import division
import os, pickle, math, random, itertools
from itertools import combinations
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA

from scipy.special import comb
import scipy.stats as stats
import networkx as nx
from asa159 import rcont2

#np.random.seed(123456789)





def get_F_stat_pairwise(pca_array, groups, k=3):
    # assuming only two groups
    # groups is a nested list, each list containing row integers for a given group
    X = pca_array[:,0:k]
    between_var = 0
    within_var = 0
    K = len(groups)
    N = np.shape(X)[0]
    euc_sq_dists = np.square(euclidean_distances(X, X))
    SS_T = sum( np.sum(euc_sq_dists, axis =1)) /  (N*2)
    euc_sq_dists_group1 = euc_sq_dists[groups[0][:, None], groups[0]]
    SS_W = sum( np.sum(euc_sq_dists_group1, axis =1)) /  (len(groups[0])*2)
    # between groups sum-of-squares
    SS_A = SS_T - SS_W


    return (SS_A / (K-1)) / (SS_W / (N-K) )





def get_ba_cov_matrix(n_genes, cov, prop=1, m=2, get_node_edge_sum=False, rho=None):
    '''Based off of Gershgorin circle theorem, we can expect
    that the code will eventually produce mostly matrices
    that aren't positive definite as the covariance value
    increases and/or more edges added to incidence matrix'''
    while True:
        if rho == None:
            ntwk = nx.barabasi_albert_graph(n_genes, m)
            ntwk_np = nx.to_numpy_matrix(ntwk)
        else:
            ntwk_np, rho_estimate = get_correlated_rndm_ntwrk(n_genes, m=m, rho=rho, count_threshold = 10000)
        C = ntwk_np * cov
        np.fill_diagonal(C, 1)
        if prop < 1:
            diag_C = np.tril(C, k =-1)
            i,j = np.nonzero(diag_C)
            ix = np.random.choice(len(i), int(np.floor((1-prop) * len(i))), replace=False)
            C[np.concatenate((i[ix],j[ix]), axis=None), np.concatenate((j[ix],i[ix]), axis=None)] = -1*cov
        if np.all(np.linalg.eigvals(C) > 0) == True:
            if rho == None:
                return C
            else:
                return C, rho_estimate
            #if get_node_edge_sum==False:
            #    return C
            #else:
            #    return C, ntwk_np.sum(axis=0)





def get_pois_sample(lambda_, u):
    x = 0
    p = math.exp(-lambda_)
    s = p
    #u = np.random.uniform(low=0.0, high=1.0)
    while u > s:
         x = x + 1
         p  = p * lambda_ / x
         s = s + p
    return x


def get_count_pop(lambdas, C):
    mult_norm = np.random.multivariate_normal(np.asarray([0]* len(lambdas)), C)#, tol=1e-6)
    mult_norm_cdf = stats.norm.cdf(mult_norm)
    counts = [ get_pois_sample(lambdas[i], mult_norm_cdf[i]) for i in range(len(lambdas))  ]

    return np.asarray(counts)


def comb_n_muts_k_genes(k, gene_sizes):
    G = len(gene_sizes)
    gene_sizes = list(gene_sizes)
    number_states = 0
    for i in range(0, len(gene_sizes) + 1):
        comb_sum = 0
        for j in list(itertools.combinations(gene_sizes, i)):
            if (len(j) > 0): #and (len(j) < G):
                s_i_j = sum( j ) + (len(j)*1)
            else:
                s_i_j = sum( j )

            comb_s_i_j = comb(N = G+k-1-s_i_j, k = G-1)
            comb_sum += comb_s_i_j

        number_states += ((-1) ** i) * comb_sum

    return number_states


def complete_nonmutator_lines():
    return ['m5','m6','p1','p2','p4','p5']


def nonmutator_shapes():
    return {'m5': 'o','m6':'s','p1':'^','p2':'D','p4':'P','p5':'X'}


def complete_mutator_lines():
    return ['m1','m4','p3']


def get_mean_centroid_distance(array, k = 3):
    X = array[:,0:k]
    #centroids = np.mean(X, axis = 0)
    return np.mean(np.sqrt(np.sum(np.square(X - np.mean(X, axis = 0)), axis=1)))

    #for row in X:
    #    centroid_distances.append(np.linalg.norm(row-centroids))
    #return np.mean(centroid_distances)



def get_mean_pairwise_euc_distance(array, k = 3):
    X = array[:,0:k]
    row_sum = np.sum( euclidean_distances(X, X), axis =1)
    return sum(row_sum) / ( len(row_sum) * (len(row_sum) -1)  )








def hellinger_transform(array):
    return np.sqrt((array.T/array.sum(axis=1)).T )
    #df = pd.read_csv(mydir + 'data/Tenaillon_et_al/gene_by_pop_delta.txt', sep = '\t', header = 'infer', index_col = 0)
    #return(df.div(df.sum(axis=1), axis=0).applymap(np.sqrt))





def get_random_matrix(c):
    #```GNU Lesser General Public License v3.0 code from https://github.com/maclandrol/FisherExact```
    # f2py -c -m asa159 asa159.f90
    #c = array
    key = np.array([False], dtype=bool)
    ierror = np.array([0], dtype=np.int32)
    sr, sc = c.sum(axis=1).astype(np.int32), c.sum(axis=0).astype(np.int32)
    nr, nc = len(sr), len(sc)
    n = np.sum(sr)
    replicate=1000
    results = np.zeros(replicate)

    seed=None
    # test to see if we can increase wkslimit for neutral sims!!!!
    #wkslimit=5000
    wkslimit=50000
    DFAULT_MAX_TOT = 5000
    # set default maxtot to wkslimit
    if wkslimit < DFAULT_MAX_TOT:
        wkslimit = 5000
    if seed is None:
        try:
            seed = random.SystemRandom().randint(1, 100000)
            seed = np.array([seed], dtype=np.int32)
        except:
            try:
                import time
                seed = int(time.time())
                seed = np.array([seed], dtype=np.int32)
            except:
                seed = 12345
                seed = np.array([seed], dtype=np.int32)

    if n < wkslimit:
        # we can just set the limit  to the table sum
        wkslimit = n
        pass
    else:
        # throw error immediately
        raise ValueError(
            "Limit of %d on the table sum exceded (%d), please increase workspace !" % (DFAULT_MAX_TOT, n))

    maxtot = np.array([wkslimit], dtype=np.int32)
    fact = np.zeros(wkslimit + 1, dtype=np.float, order='F')
    observed = np.zeros((nr, nc), dtype=np.int32, order='F')

    rcont2(nrow=nr, ncol=nc, nrowt=sr, ncolt=sc, maxtot=maxtot,
           key=key, seed=seed, fact=fact, matrix=observed, ierror=ierror)

    # if we do not have an error, make spcial action
    #ans = 0.
    tmp_observed = observed.ravel()
    if ierror[0] in [1, 2]:
        raise ValueError(
            "Error in rcont2 (fortran) : row or column input size is less than 2!")
    elif ierror[0] in [3, 4]:
        raise ValueError(
            "Error in rcont2 (fortran) : Negative values in table !")
    elif ierror[0] == 6:
        # this shouldn't happen with the previous check
        raise ValueError(
            "Error in rcont2 (fortran) : Limit on the table sum (%d) exceded, please increase workspace !" % DFAULT_MAX_TOT)
    else:
        return np.reshape(tmp_observed, (nr,nc))









def get_x_stat(e_values):

    def get_n_prime(e_values):
        # moments estimator from Patterson et al 2006
        # equation 10
        m = len(e_values) + 1
        sq_sum_ev = sum(e_values) ** 2
        sum_sq_ev = sum( e **2 for e in  e_values )
        return ((m+1) * sq_sum_ev) /  (( (m-1)  * sum_sq_ev ) -  sq_sum_ev )

    def get_mu(m, n):
        return ((np.sqrt(n-1) + np.sqrt(m)) ** 2) / n

    def get_sigma(m, n):
        return ((np.sqrt(n-1) + np.sqrt(m)) / n) * np.cbrt((1/np.sqrt(n-1)) + (1/np.sqrt(m)))

    def get_l(e_values):
        return (len(e_values) * max(e_values)) / sum(e_values)

    n = get_n_prime(e_values)
    m = len(e_values) + 1

    return (get_l(e_values) - get_mu(m, n)) / get_sigma(m, n)




def get_correlated_rndm_ntwrk(n_genes, m=2, rho=0.3, count_threshold = 10000):
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
        graph = nx.barabasi_albert_graph(n_genes, m)
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

    return iter_graph, iter_rh0





def get_mean_center(array):
    return array - np.mean(array, axis=0)

def matrix_vs_null_one_treat(count_matrix, iter):
    X = get_mean_center(count_matrix)
    pca = PCA()
    pca_fit = pca.fit_transform(X)
    euc_dist = get_mean_pairwise_euc_distance(pca_fit)
    euc_dists = []
    for j in range(iter):
        #X_j = pt.hellinger_transform(pt.get_random_matrix(test_cov))
        X_j = get_mean_center(get_random_matrix(count_matrix))
        pca_fit_j = pca.fit_transform(X_j)
        euc_dists.append( get_mean_pairwise_euc_distance(pca_fit_j) )
    euc_percent = len( [k for k in euc_dists if k < euc_dist] ) / len(euc_dists)
    z_score = (euc_dist - np.mean(euc_dists)) / np.std(euc_dists)
    return euc_percent, z_score


#class likelihood_matrix:
#    def __init__(self, df, dataset):
#        self.df = df.copy()
#        self.dataset = dataset
#
#    def get_gene_lengths(self, **keyword_parameters):
#        if self.dataset == 'Good_et_al':
#            conv_dict = cd.good_et_al().parse_convergence_matrix(get_path() + "/data/Good_et_al/gene_convergence_matrix.txt")
#            length_dict = {}
#            if ('gene_list' in keyword_parameters):
#                for gene_name in keyword_parameters['gene_list']:
#                    length_dict[gene_name] = conv_dict[gene_name]['length']
#                #for gene_name, gene_data in conv_dict.items():
#            else:
#                for gene_name, gene_data in conv_dict.items():
#                    length_dict[gene_name] = conv_dict[gene_name]['length']
#            return(length_dict)
#
#        elif self.dataset == 'Tenaillon_et_al':
#            with open(get_path() + '/data/Tenaillon_et_al/gene_size_dict.txt', 'rb') as handle:
#                length_dict = pickle.loads(handle.read())
#                if ('gene_list' in keyword_parameters):
#                    return { gene_name: length_dict[gene_name] for gene_name in keyword_parameters['gene_list'] }
#                    #for gene_name in keyword_parameters['gene_list']:
#                else:
#                    return(length_dict)

#    def get_likelihood_matrix(self):
#        genes = self.df.columns.tolist()
#        genes_lengths = self.get_gene_lengths(gene_list = genes)
#        L_mean = np.mean(list(genes_lengths.values()))
#        L_i = np.asarray(list(genes_lengths.values()))
#        N_genes = len(genes)
#        m_mean = self.df.sum(axis=1) / N_genes

#        for index, row in self.df.iterrows():
#            m_mean_j = m_mean[index]
#            np.seterr(divide='ignore')
#            delta_j = row * np.log((row * (L_mean / L_i)) / m_mean_j)
#            self.df.loc[index,:] = delta_j

#        df_new = self.df.fillna(0)
#        # remove colums with all zeros
#        df_new.loc[:, (df_new != 0).any(axis=0)]
#        # replace negative values with zero
#        df_new[df_new < 0] = 0
#        return df_new
