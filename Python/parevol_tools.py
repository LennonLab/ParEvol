from __future__ import division
import os, pickle, math, random, itertools
from itertools import combinations
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA, SparsePCA
from scipy.linalg import block_diag
from scipy.special import comb
import scipy.stats as stats
import networkx as nx
from asa159 import rcont2
from copy import copy
import matplotlib.colors as cls

#np.random.seed(123456789)

def get_alpha():
    return 0.05



class classFASTA:

    def __init__(self, fileFASTA):
        self.fileFASTA = fileFASTA

    def readFASTA(self):
        '''Checks for fasta by file extension'''
        file_lower = self.fileFASTA.lower()
        '''Check for three most common fasta file extensions'''
        if file_lower.endswith('.txt') or file_lower.endswith('.fa') or \
        file_lower.endswith('.fasta') or file_lower.endswith('.fna') or \
        file_lower.endswith('.faa') or file_lower.endswith('.ffn'):
            with open(self.fileFASTA, "r") as f:
                return self.ParseFASTA(f)
        else:
            print("Not in FASTA format.")

    def ParseFASTA(self, fileFASTA):
        '''Gets the sequence name and sequence from a FASTA formatted file'''
        fasta_list=[]
        for line in fileFASTA:
            if line[0] == '>':
                try:
                    fasta_list.append(current_dna)
            	#pass if an error comes up
                except UnboundLocalError:
                    #print "Inproper file format."
                    pass
                current_dna = [line.lstrip('>').rstrip('\n'),'']
            else:
                current_dna[1] += "".join(line.split())
        fasta_list.append(current_dna)
        '''Returns fasa as nested list, containing line identifier \
            and sequence'''
        return fasta_list


def get_mean_colors(c1, c2, w1, w2):
    # c1 and c2 are in hex format
    # w1 and w2 are the weights
    c1_list = list(cls.to_rgba('#FF3333'))
    c2_list = list(cls.to_rgba('#3333FF'))
    zipped = list(zip(c1_list, c2_list))
    new_rgba = []
    for item in zipped:
        new_rgba.append(math.exp((w1 * math.log(item[0])) + (w2 * math.log(item[1]))))
    #weight_sum = w1 + w2
    return cls.rgb2hex(tuple(new_rgba))



def get_ba_cov_matrix(n_genes, cov, cov2 = None, prop=False, m=2, get_node_edge_sum=False, rho=None, rho2=None):
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

        if cov2 == None:
            C = ntwk_np * cov
        else:
            C = ntwk_np * cov
            C2 = ntwk_np * cov2
            np.fill_diagonal(C2, 1)

        np.fill_diagonal(C, 1)

        if prop == True and cov2 == None:
            diag_C = np.tril(C, k =-1)
            i,j = np.nonzero(diag_C)
            ix = np.random.choice(len(i), int(np.floor((1-prop) * len(i))), replace=False)
            C[np.concatenate((i[ix],j[ix]), axis=None), np.concatenate((j[ix],i[ix]), axis=None)] = -1*cov

        if cov2 == None:
            if np.all(np.linalg.eigvals(C) > 0) == True:
                if rho == None:
                    return C
                else:
                    return C, rho_estimate

        else:
            if (np.all(np.linalg.eigvals(C) > 0) == True) and (np.all(np.linalg.eigvals(C2) > 0) == True):
                return C, C2






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



def get_correlated_rndm_ntwrk(n_genes, m=2, rho=0.3, rho2=None, count_threshold = 10000, rho_error = 0.01):
    #  Xalvi-Brunet and Sokolov
    # generate maximally correlated networks with a predefined degree sequence
    # assumes that abs(rho) > abs(rho2)
    if rho > 0:
        assortative = True
    elif rho <= 0:
        assortative = False

    if rho2 != None:
        if rho2 > rho:
            assortative2 = True
        elif rho2 <= rho:
            assortative2 = False

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
        # for some reason sometimes np.sum() returns a nested list?
        # check for that
        #row_sums = np.asarray(np.sum(graph_array, axis =0))[0]
        row_sums = np.asarray(np.sum(graph_array, axis =0)).tolist()
        if any(isinstance(i, list) for i in row_sums) == True:
            row_sums = [item for sublist in row_sums for item in sublist]
        else:
            row_sums = row_sums
        node_edge_counts = [(l0_n0, row_sums[l0_n0]), (l0_n1, row_sums[l0_n1]),
                            (link1[0], row_sums[link1[0]]), (link1[1], row_sums[link1[1]])]
        return node_edge_counts

    if rho == 0:
        iter_rho = 2*rho_error
    else:
        iter_rho = 0
    iter_graph = None

    #while (assortative == True and iter_rho < rho) or (assortative == False and iter_rho > rho):
    while (iter_rho > rho + rho_error) or (iter_rho < rho - rho_error):
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

        iter_rho = nx.degree_assortativity_coefficient(nx.from_numpy_matrix(graph_np))
        iter_graph = graph_np


    if rho2 == None:
        return iter_graph, iter_rho

    else:
        iter_rho2 = copy(iter_rho)
        graph_np_2 = np.copy(graph_np)
        current_rho2 = copy(iter_rho)
        accepted_counts2 = 0
        count2 = 0

        # iter_rho > rho + rho_error) or (iter_rho < rho - rho_error
        #while ((assortative2 == True and current_rho2 < rho2) or (assortative2 == False and current_rho2 > rho2)) and ((count2-accepted_counts2) < count_threshold):
        while ((assortative2 == True and current_rho2 < rho2 - rho_error) or (assortative2 == False and current_rho2 > rho2 + rho_error)) and ((count2-accepted_counts2) < count_threshold):
            count2 += 1
            edges = get_two_edges(graph_np_2)
            graph_np_sums2 = np.sum(graph_np_2, axis=1)
            # check whether new edges already exist
            if graph_np_2[edges[0][0],edges[3][0]] == 1 or \
                graph_np_2[edges[3][0],edges[0][0]] == 1 or \
                graph_np_2[edges[2][0],edges[1][0]] == 1 or \
                graph_np_2[edges[1][0],edges[2][0]] == 1:
                continue

            disc = (edges[0][1] - edges[2][1]) * \
                    (edges[3][1] - edges[1][1])
            if (assortative2 == True and disc > 0) or (assortative2 == False and disc < 0):
                graph_np_2[edges[0][0],edges[1][0]] = 0
                graph_np_2[edges[1][0],edges[0][0]] = 0
                graph_np_2[edges[2][0],edges[3][0]] = 0
                graph_np_2[edges[3][0],edges[2][0]] = 0

                graph_np_2[edges[0][0],edges[3][0]] = 1
                graph_np_2[edges[3][0],edges[0][0]] = 1
                graph_np_2[edges[2][0],edges[1][0]] = 1
                graph_np_2[edges[1][0],edges[2][0]] = 1

                accepted_counts2 += 1
                current_rho2 = nx.degree_assortativity_coefficient(nx.from_numpy_matrix(graph_np_2))

        iter_rho2 = nx.degree_assortativity_coefficient(nx.from_numpy_matrix(graph_np_2))
        iter_graph2 = graph_np_2

        return iter_graph, iter_rho, iter_graph2, iter_rho2



def get_mean_center(array):
    return array - np.mean(array, axis=0)


def matrix_vs_null_one_treat(count_matrix, iter):
    count_matrix_rel = count_matrix/count_matrix.sum(axis=1)[:,None]
    X = get_mean_center(count_matrix_rel)
    pca = PCA()
    #pca = SparsePCA(normalize_components=True)
    pca_fit = pca.fit_transform(X)
    euc_dist = get_mean_pairwise_euc_distance(pca_fit)
    euc_dists = []
    for j in range(iter):
        #X_j = pt.hellinger_transform(pt.get_random_matrix(test_cov))
        count_matrix_j = get_random_matrix(count_matrix)
        count_matrix_rel_j = count_matrix_j/count_matrix_j.sum(axis=1)[:,None]
        X_j = get_mean_center(count_matrix_rel_j)
        pca_fit_j = pca.fit_transform(X_j)
        euc_dists.append( get_mean_pairwise_euc_distance(pca_fit_j) )
    euc_percent = len( [k for k in euc_dists if k < euc_dist] ) / len(euc_dists)
    z_score = (euc_dist - np.mean(euc_dists)) / np.std(euc_dists)
    return euc_percent, z_score


def get_F_2(count_matrix, N1, N2):
    '''
    Modified F-statistic from Anderson et al., 2017 doi: 10.1111/anzs.12176
    Function assumes that the rows of the count matrix are sorted by group
    i.e., group one is first N1 rows, rest of the N2 rows are group two
    '''
    N = N1+N2
    count_matrix_rel = count_matrix/count_matrix.sum(axis=1)[:,None]
    X = get_mean_center(count_matrix_rel)
    pca = PCA()
    pca_fit = pca.fit_transform(X)
    dist_matrix = euclidean_distances(X, X)
    A = -(1/2) * (dist_matrix ** 2)
    I = np.identity(N)
    J_N = np.full((N, N), 1)
    G = (I - ((1/N) * J_N )) @ A @ (I - ((1/N) * J_N ))
    n1 = (1/N1) * np.full((N1, N1), 1)
    n2 = (1/N2) * np.full((N2, N2), 1)
    H = block_diag(n1, n2) - ((1/N) * J_N )
    # indicator matrices
    U_1 = np.diag( (N1*[1]) + (N2*[0]))
    U_2 = np.diag( (N1*[0]) + (N2*[1]))

    V_1 = np.trace(((I - H) @ U_1 @ (I - H)) @ G ) / (N1-1)
    V_2 = np.trace(((I - H) @ U_2 @ (I - H)) @ G ) / (N2-1)

    F_2 = np.trace(H @ G) / (((1- (N1/N)) * V_1) + ((1- (N2/N)) * V_2))

    return F_2, V_1, V_2


def matrix_vs_null_two_treats(count_matrix,  N1, N2, iter=1000):
    F_2, V_1, V_2 = get_F_2(count_matrix, N1, N2)
    F_2_list = []
    V_1_list = []
    V_2_list = []
    for j in range(iter):
        F_2_j, V_1_j, V_2_j = get_F_2(get_random_matrix(count_matrix), N1, N2)
        F_2_list.append(F_2_j)
        V_1_list.append(V_1_j)
        V_2_list.append(V_2_j)
    F_2_percent = len( [k for k in F_2_list if k < F_2] ) / iter
    F_2_z_score = (F_2 - np.mean(F_2_list)) / np.std(F_2_list)
    V_1_percent = len( [k for k in V_1_list if k < V_1] ) / iter
    V_1_z_score = (V_1 - np.mean(V_1_list)) / np.std(V_1_list)
    V_2_percent = len( [k for k in V_2_list if k < V_2] ) / iter
    V_2_z_score = (V_2 - np.mean(V_2_list)) / np.std(V_2_list)

    return F_2_percent, F_2_z_score, V_1_percent, V_1_z_score, V_2_percent, V_2_z_score
