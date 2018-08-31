from __future__ import division
import os, pickle, math, random, itertools
from itertools import combinations
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import matplotlib.colors as cls
import rpy2.robjects as robjects
import clean_data as cd
#import scipy.spatial.distance as dist
from scipy.spatial.distance import pdist, squareform
from scipy.misc import comb
from scipy.special import binom
from statsmodels.base.model import GenericLikelihoodModel
import networkx as nx

#np.random.seed(123456789)


def get_path():
    return os.path.expanduser("~/GitHub/ParEvol")

def get_bray_curtis(array):
    distance_array = np.zeros((array.shape[0], array.shape[0]))
    for i, row_i in enumerate(array):
        for j, row_j in enumerate(array):
            if i <= j:
                continue
            C_ij =  sum([min(x) for x in list(zip(row_i, row_j))])
            S_i = np.sum(row_i)
            S_j = np.sum(row_j)
            BC_ij = 1 - ((2*C_ij) / (S_i + S_j))
            distance_array[i,j] = distance_array[j,i] = BC_ij
    return distance_array


def get_adjacency_matrix(array):
    array = np.transpose(array)
    adjacency_array = np.zeros((array.shape[0], array.shape[0]))
    for i, row_i in enumerate(array):
        for j, row_j in enumerate(array):
            if i <= j:
                continue
            test = [1 if ((x[0] > 0) and (x[1] > 0)) else 0 for x in list(zip(row_i, row_j))  ]
            if sum(test) > 0:
                adjacency_array[i,j] = adjacency_array[j,i] = 1
            else:
                adjacency_array[i,j] = adjacency_array[j,i] = 0

    return adjacency_array


def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

#def get_multiplicity_dist(m):

def comb_n_muts_k_genes(n, gene_sizes):
    #n = 7
    #gene_sizes = [2, 3, 4]
    k = len(gene_sizes)
    def findsubsets(S,m):
        return set(itertools.combinations(S, m))
    B = []
    for count in range(0, len(gene_sizes) + 1):
        for subset in findsubsets(set(gene_sizes), count):
            B.append(list(subset))
    number_ways = 0
    for S in B:
        n_S = n + k - 1 - (sum(S) + (1 * len(S) ) )
        if n_S < (k-1):
            continue

        number_ways +=  ((-1) ** len(S)) * comb(N = n_S, k = k-1)
    return number_ways




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


'''code is from https://stackoverflow.com/questions/6284396/permutations-with-unique-values'''

class unique_element:
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences

def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1

def perm_unique(elements):
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)




def partition(lst, n):
    # partitions a list into n lists of equal length
    random.shuffle(lst)
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]



def get_scipy_bray_curtis(array):
    return squareform(pdist(array, metric = 'braycurtis'))


def complete_nonmutator_lines():
    return ['m5','m6','p1','p2','p4','p5']

def nonmutator_shapes():
    return {'m5': 'o','m6':'s','p1':'^','p2':'D','p4':'P','p5':'X'}


def complete_mutator_lines():
    return ['m1','m4','p3']


def cmdscale(D):
    """
    Classical multidimensional scaling (MDS)

    Parameters
    ----------
    D : (n, n) array
        Symmetric distance matrix.

    Returns
    -------
    Y : (n, p) array
        Configuration matrix. Each column represents a dimension. Only the
        p dimensions corresponding to positive eigenvalues of B are returned.
        Note that each dimension is only determined up to an overall sign,
        corresponding to a reflection.

    e : (n,) array
        Eigenvalues of B.
    Acquired from http://www.nervouscomputer.com/hfs/cmdscale-in-python/
    """
    # Number of points
    n = len(D)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n))/n

    # YY^T
    B = -H.dot(D**2).dot(H)/2

    print(B.T * B)

    # Diagonalize
    evals, evecs = np.linalg.eigh(B)

    # Sort by eigenvalue in descending order
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)

    return Y, evals

def get_pcoa(df):
    # remove columns containing only zeros
    df_no0 = df.loc[:, (df != 0).any(axis=0)]
    # only keep pops from day 100
    ids = df_no0.index.values
    data = df_no0.values
    ds = get_ds(data)
    pcoa = cmdscale(ds)
    Y = pd.DataFrame(pcoa[0])
    Y['pops'] = ids
    Y = Y.set_index('pops')
    return([Y, pcoa[1]])

def get_mean_centroid_distance(array, groups = None, k = 3):
    X = array[:,0:k]
    centroid_distances = []
    centroids = np.mean(X, axis = 0)
    for row in X:
        centroid_distances.append(np.linalg.norm(row-centroids))
    return np.mean(centroid_distances)

def get_euclidean_distance(array, k = 3):
    X = array[:,0:k]
    rows = list(range(array.shape[0]))
    angle_pairs = []
    for i in rows:
        for j in rows:
            if i < j:
                row_i = X[i,:]
                row_j = X[j,:]
                # difference in magnitude
                angle_pairs.append( abs(np.linalg.norm(row_i) - np.linalg.norm(row_j)) )

    return (sum(angle_pairs) * 2) / (len(rows) * (len(rows)-1))



def get_mean_angle(array, k = 3):
    def angle_between(v1, v2):
        radians = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return radians * 180 / math.pi

    X = array[:,0:k]
    rows = list(range(array.shape[0]))
    angle_pairs = []
    for i in rows:
        for j in rows:
            if i < j:
                row_i = X[i,:]
                row_j = X[j,:]
                angle_pairs.append( angle_between(row_i, row_j) )

    return (sum(angle_pairs) * 2) / (len(rows) * (len(rows)-1))







def hellinger_transform(array):
    return np.sqrt((array.T/array.sum(axis=1)).T )
    #df = pd.read_csv(mydir + 'data/Tenaillon_et_al/gene_by_pop_delta.txt', sep = '\t', header = 'infer', index_col = 0)
    #return(df.div(df.sum(axis=1), axis=0).applymap(np.sqrt))


def random_matrix(array):
    ### use
    ###  switch to ASA159 algorithm
    r2dtable = robjects.r['r2dtable']
    row_sum = array.sum(axis=1)
    column_sum = array.sum(axis=0)
    sample = r2dtable(1, robjects.IntVector(row_sum), robjects.IntVector(column_sum))
    return np.asarray(sample[0])



def get_broken_stick(array):
    # Legendre & Legendre, eqn. 9.16
    array = np.sort(array)
    out_list = []
    for j in range(1, len(array)+1):
        #print(sum((1/x) for x in range(j, len(array)) ))
        out_list.append(sum((1/x) for x in range(j, len(array)+1) ))
    return np.asarray(out_list) * (1 / len(array))


def plot_eigenvalues(explained_variance_ratio_, file_name = 'eigen'):
    x = range(1, len(explained_variance_ratio_) + 1)
    if sum(explained_variance_ratio_) != 1:
        y = explained_variance_ratio_ / sum(explained_variance_ratio_)
    else:
        y = explained_variance_ratio_
    y_bs = get_broken_stick(explained_variance_ratio_)

    fig = plt.figure()
    plt.plot(x, y_bs, marker='o', linestyle='--', color='r', label='Broken-stick',markeredgewidth=0.0, alpha = 0.6)
    plt.plot(x, y, marker='o', linestyle=':', color='k', label='Observed', markeredgewidth=0.0, alpha = 0.6)
    plt.xlabel('PCoA axis', fontsize = 16)
    plt.ylabel('Percent vaiance explained', fontsize = 16)

    fig.tight_layout()
    out_path = get_path() + '/figs/' + file_name + '.png'
    fig.savefig(out_path, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


class likelihood_matrix:
    def __init__(self, df, dataset):
        self.df = df
        self.dataset = dataset

    def get_gene_lengths(self, **keyword_parameters):
        if self.dataset == 'Good_et_al':
            conv_dict = cd.good_et_al().parse_convergence_matrix(get_path() + "/data/Good_et_al/gene_convergence_matrix.txt")
            length_dict = {}
            if ('gene_list' in keyword_parameters):
                for gene_name in keyword_parameters['gene_list']:
                    length_dict[gene_name] = conv_dict[gene_name]['length']
                #for gene_name, gene_data in conv_dict.items():
            else:
                for gene_name, gene_data in conv_dict.items():
                    length_dict[gene_name] = conv_dict[gene_name]['length']
            return(length_dict)

        elif self.dataset == 'Tenaillon_et_al':
            with open(get_path() + '/data/Tenaillon_et_al/gene_size_dict.txt', 'rb') as handle:
                length_dict = pickle.loads(handle.read())
                return(length_dict)

    def get_likelihood_matrix(self):
        #df_in = get_path() + '/data/' + self.dataset + '/gene_by_pop.txt'
        #df = pd.read_csv(df_in, sep = '\t', header = 'infer', index_col = 0)
        genes = self.df.columns.tolist()
        genes_lengths = self.get_gene_lengths(gene_list = genes)
        L_mean = np.mean(list(genes_lengths.values()))
        L_i = np.asarray(list(genes_lengths.values()))
        N_genes = len(genes)
        m_mean = self.df.sum(axis=1) / N_genes

        for index, row in self.df.iterrows():
            m_mean_j = m_mean[index]
            delta_j = row * np.log((row * (L_mean / L_i)) / m_mean_j)
            self.df.loc[index,:] = delta_j

        #out_name = get_path() + '/data/' + self.dataset + '/gene_by_pop_delta.txt'

        df_new = self.df.fillna(0)
        # remove colums with all zeros
        df_new.loc[:, (df_new != 0).any(axis=0)]
        # replace negative values with zero
        df_new[df_new < 0] = 0
        return df_new
        #df_new.to_csv(out_name, sep = '\t', index = True)


# Power law
def power_law(t, c0, v):
    t = np.asarray(t)
    return c0 * (t ** (1 / (v-1)) )



class modifiedGompertz(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(modifiedGompertz, self).__init__(endog, exog, **kwds)
        #print len(exog)

    def nloglikeobs(self, params):
        c0 = params[0]
        v = params[1]
        # probability density function (pdf) is the same as dnorm
        exog_pred = m_gop(self.endog, b0 = b0, A = A, umax = umax, L = L)
        # need to flatten the exogenous variable
        LL = -stats.norm.logpdf(self.exog.flatten(), loc=exog_pred, scale=np.exp(z))
        return LL

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method="bfgs", **kwds):

        if start_params is None:
            b0_start = 1
            A_start = 2
            umax_start = 0.5
            L_start = 0.8
            z_start = 0.8

            start_params = np.array([b0_start, A_start, umax_start, L_start, z_start])

        return super(modifiedGompertz, self).fit(start_params=start_params,
                                maxiter=maxiter, method = method, maxfun=maxfun,
                                **kwds)

def get_random_network_probability(df):
    # df is a numpy matrix or pandas dataframe containing network interactions
    N = df.shape[0]
    M = 0
    k_list = []
    for index, row in df.iterrows():
        # != 0 bc there's positive and negative interactions
        k_row = sum(i != 0 for i in row.values) - 1
        if k_row > 0:
            k_list.append(k_row)
    # run following algorithm to generate random network
    # (1) Start with N isolated nodes.
    # (2) Select a node pair and generate a random number between 0 and 1.
    #     If the number exceeds p, connect the selected node pair with a link,
    #     otherwise leave them disconnected.
    # (3) Repeat step (2) for each of the N(N-1)/2 node pairs.
    M = sum(k_list)
    p = M / binom(N, 2)
    matrix = np.ones((N,N))
    #node_pairs = list(combinations(range(int((N * (N-1)) / 2)), 2))
    node_pairs = list(combinations(range(N), 2))
    for node_pair in node_pairs:
        p_node_pair = random.uniform(0, 1)
        if p_node_pair > p:
            continue
        else:
            matrix[node_pair[0], node_pair[1]] = 0
            matrix[node_pair[1], node_pair[0]] = 0

    return matrix

def get_random_network_edges(df):
    nodes = df.index.tolist()
    edges = []
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i < j:
                edges.append(node_i + '-' + node_j)
    L = 0
    for index, row in df.iterrows():
        #k_row = sum(float(k) != float(0) for k in row.values) - 1
        k_row = len([i for i in row.values if i != 0]) - 1
        L += k_row

    new_edges = np.random.choice(np.asarray(edges), size=int(L/2), replace=False)
    new_edges_split = [x.split('-') for x in new_edges]
    matrix = pd.DataFrame(0, index= nodes, columns=nodes)
    for node in nodes:
        matrix.loc[node, node] = 1
    for new_edge in new_edges_split:
        matrix.loc[new_edge[0], new_edge[1]] = 1
        matrix.loc[new_edge[1], new_edge[0]] = 1

    L_test = 0
    for index_m, row_m in matrix.iterrows():
        k_row_m = len([i for i in row_m.values if i != 0]) - 1
        L_test += k_row_m

    return matrix


def networkx_distance(df):
    def get_edges(nodes):
        edges = []
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i < j:
                    pair_value = df.loc[node_i][node_j]
                    if pair_value > 0:
                        edges.append((node_i,node_j))
        return(edges)

    edges_full = get_edges(df.index.tolist())
    graph = nx.Graph()
    graph.add_edges_from(edges_full)

    if nx.is_connected(graph) == True:
        return(nx.average_shortest_path_length(graph))
    else:
        components = [list(x) for x in nx.connected_components(graph)]
        component_distances = []
        # return a grand mean if there are seperate graphs
        for component in components:
            component_edges = get_edges(component)
            graph_component = nx.Graph()
            graph_component.add_edges_from(component_edges)
            component_distance = nx.average_shortest_path_length(graph_component)
            component_distances.append(component_distance)
        return( np.mean(component_distances) )

        #print(components)
