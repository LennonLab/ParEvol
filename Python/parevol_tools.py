from __future__ import division
import os, pickle, math, random, itertools
from itertools import combinations
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA

from scipy.linalg import block_diag
from scipy.special import comb
import scipy.stats as stats
from scipy.special import gammaln
from scipy import linalg as LA

import networkx as nx
from asa159 import rcont2
from copy import copy
import matplotlib.colors as cls

np.random.seed(123456789)

def get_alpha():
    return 0.05


# calculate_total_parallelism function is modified from GitHub repo
# benjaminhgood/LTEE-metagenomic under GPL v2
def calculate_total_parallelism(gene_statistics, allowed_genes=None, num_bootstraps=10000):

    if allowed_genes==None:
        allowed_genes = gene_statistics.keys()

    Ls = []
    ns = []

    for gene_name in allowed_genes:

        Ls.append( gene_statistics[gene_name]['length'] )
        ns.append( gene_statistics[gene_name]['observed'] )


    Ls = np.array(Ls)
    ns = np.array(ns)

    Ltot = Ls.sum()
    ntot = ns.sum()
    ps = Ls*1.0/Ltot

    gs = ns*np.log(ns/(ntot*ps)+(ns==0))

    observed_G = gs.sum()/ns.sum()
    #bootstrapped_Gs = []
    #for bootstrap_idx in range(0,num_bootstraps):
    #    bootstrapped_ns = np.random.multinomial(ntot,ps)
    #    bootstrapped_gs = bootstrapped_ns*np.log(bootstrapped_ns/(ntot*ps)+(bootstrapped_ns==0))
    #    bootstrapped_G = bootstrapped_gs.sum()/bootstrapped_ns.sum()

    #    bootstrapped_Gs.append(bootstrapped_G)

    #bootstrapped_Gs = np.array(bootstrapped_Gs)

    #pvalue = ((bootstrapped_Gs>=observed_G).sum()+1.0)/(len(bootstrapped_Gs)+1.0)
    return observed_G#, pvalue



# code from https://github.com/lubeme/Scale-Free-Random-Walks-Networks
def random_walks_powerlaw_cluster_graph(m,n,cc,seed=None):
    """Return random graph using Herrera-Zufiria random walks model.

    A Scale-free graph of n nodes is grown by attaching new nodes
    each with m edges that are connected to existing by performing
    random walks,using only local information of the network.
    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    cc: int
        clustering control parameter, from 0 to 100. Increasing control
        parameter implies increasing the clustering of the graph
    seed : int, optional
        Seed for random number generator (default=None).
    Returns
    -------
    G : Graph

    Notes
    -----
    The initialization is a circular graph with an odd number of nodes.
    For small values of m initial graph has at least 11 nodes.
    References
    ----------
    .. [1] Herrera, C.; Zufiria, P.J.; , "Generating scale-free networks
    with adjustable clustering coefficient via random walks,"
    Network Science Workshop (NSW),
    2011 IEEE , vol., no., pp.167-172, 22-24 June 2011
    """


    if m < 1 or  m >=n or cc < 0 or cc > 100:
        raise nx.NetworkXError(\
            "The network must have m>=1, m<n and "
            "cc between 0 and 100. m=%d,n=%d cc=%d"%(m,n,cc))

    if seed is not None:
        random.seed(seed)

    nCero= max(11,m)
    if nCero%2==0:
        nCero+=1
    #initialise graph
    G= nx.generators.classic.cycle_graph(nCero)
    G.name="Powerlaw-Cluster Random-Walk Graph"

    #list of probabilities 'pi' associated to each node
    #representing a genetic factor
    p=stats.bernoulli.rvs(cc/float(100),size=n)

    #main loop of the algorithm
    for j in range(nCero,n):
        #Choose Random node
        vs =random.randrange(0,G.number_of_nodes())
        #random walk of length>1 beginning on vs
        l = 7
        ve=vs
        for i in range(l):
            neighborsVe = G.neighbors(ve)
            neighborsVe_list = list(neighborsVe)
            # random.choice(numberList)
            #ve= list(neighborsVe)[random.randrange(0,len(list(neighborsVe)))]
            ve = random.choice(neighborsVe_list)
            #len(list(somegraph.neighbors(somenode)))


        markedVertices=[]
        #mark ve
        markedVertices.append(ve)
        vl=ve
        #Mark m nodes
        for i in range(m-1):
            #Random walk of l = [1 , 2] depending on the
            #genetic factor of the node vl
            l =2 - p[vl]
            vll=vl
            #Random Walk starting on vl, avoiding already marked vertices
            while ((vll in markedVertices)):
                print(vll)
                for k in range(l):
                    neighborsVl = G.neighbors(vll)
                    neighborsVl_list = list(neighborsVl)
                    #vll= neighborsVl[random.randrange(0,len(neighborsVl))]
                    vll = random.choice(neighborsVl_list)
            vl=vll
            #mark vl
            markedVertices.append(vl)
        #Add the new node
        G.add_node(j)
        #Assign the node a pi
        #Add the m marked neighbors to vl
        for i in range(m):
            G.add_edge(j,markedVertices[i])
    return G



# calculate_poisson_log_survival function is modified from GitHub repo
# benjaminhgood/LTEE-metagenomic under GPL v2
def calculate_poisson_log_survival(ns, expected_ns):
    # change threshold from 1e-20 to 1e-60 so that genes w/ most mutations can pass

    survivals = stats.poisson.sf(ns-0.1, expected_ns)

    logsurvivals = np.zeros_like(survivals)
    logsurvivals[survivals>1e-60] = -np.log(survivals[survivals>1e-60])
    logsurvivals[survivals<=1e-60] = (-ns*np.log(ns/expected_ns+(ns==0))+ns-expected_ns)[survivals<=1e-60]

    return logsurvivals


# calculate_parallelism_logpvalues function is modified from GitHub repo
# benjaminhgood/LTEE-metagenomic under GPL v2
def calculate_parallelism_logpvalues(gene_statistics):

    gene_names = []
    Ls = []
    ns = []
    expected_ns = []

    for gene_name in gene_statistics.keys():
        gene_names.append(gene_name)
        ns.append(gene_statistics[gene_name]['observed'])
        expected_ns.append(gene_statistics[gene_name]['expected'])

    ns = np.array(ns)
    expected_ns = np.array(expected_ns)


    logpvalues = calculate_poisson_log_survival(ns, expected_ns)

    return {gene_name: logp for gene_name, logp in zip(gene_names, logpvalues)}


# NullGeneLogpSurvivalFunction class is modified from GitHub repo
# benjaminhgood/LTEE-metagenomic under GPL v2
class NullGeneLogpSurvivalFunction(object):
    # Null distribution of -log p for each gene

    def __init__(self, Ls, ntot,nmin=0):
        self.ntot = ntot
        self.Ls = np.array(Ls)*1.0
        self.Lavg = self.Ls.mean()
        self.ps = self.Ls/self.Ls.sum()
        self.expected_ns = self.ntot*self.ps
        self.nmin = nmin

    @classmethod
    def from_parallelism_statistics(cls, gene_parallelism_statistics,nmin=0):

        # calculate Ls
        Ls = []
        ntot = 0
        for gene_name in gene_parallelism_statistics.keys():
            Ls.append(gene_parallelism_statistics[gene_name]['length'])
            ntot += gene_parallelism_statistics[gene_name]['observed']

        return cls(Ls, ntot, nmin)

    def __call__(self, mlogps):

        # Do sum by hand
        ns = np.arange(0,400)*1.0

        logpvalues = calculate_poisson_log_survival(ns[None,:], self.expected_ns[:,None])

        logprobabilities = ns[None,:]*np.log(self.expected_ns)[:,None]-gammaln(ns+1)[None,:]-self.expected_ns[:,None]
        probabilities = np.exp(logprobabilities)
        survivals = np.array([ ((logpvalues>=mlogp)*(ns[None,:]>=self.nmin)*probabilities).sum() for mlogp in mlogps])
        return survivals



# calculate_unnormalized_survival_from_vector function is modified from GitHub repo
# benjaminhgood/LTEE-metagenomic under GPL v2
def calculate_unnormalized_survival_from_vector(xs, min_x=None, max_x=None, min_p=1e-10):
    if min_x==None:
        min_x = xs.min()-1

    if max_x==None:
        max_x = xs.max()+1

    unique_xs = set(xs)
    unique_xs.add(min_x)
    unique_xs.add(max_x)

    xvalues = []
    num_observations = []

    for x in sorted(unique_xs):
        xvalues.append(x)
        num_observations.append( (xs>=x).sum() )

    # So that we can plot CDF, SF on log scale
    num_observations[0] -= min_p
    num_observations[1] -= min_p
    num_observations[-1] += min_p

    return np.array(xvalues), np.array(num_observations)



def get_mean_center(array):
    return array - np.mean(array, axis=0)


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def get_mean_pairwise_euc_distance(array, k = 3):
    #X = array[0:k,:]
    X = array[:,:k]
    row_sum = np.sum( euclidean_distances(X, X), axis =1)
    return sum(row_sum) / ( len(row_sum) * (len(row_sum) -1)  )



def pca_np(x):
    # mean center matrix
    x -= np.mean(x, axis = 0)
    #print(x)
    cov = np.cov(x, rowvar = False)
    evals , evecs = LA.eigh(cov)

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]

    a = np.dot(x, evecs)


    return evals, a

    #X = obs_matrix
    #X = X[:, (X != 0).sum(axis=0) > 1]
    #n = X.shape[0]
    #p = X.shape[1]
    # covariance matrix
    # center observations on the mean
    #C = np.identity(n) - (1/n) * np.full((n,n), 1)
    #X_C = np.dot(C, X)
    #S = np.dot(X_C, X_C.T) / (n-1)

    # Eigen decomposition
    #e_vals, e_vecs = np.linalg.eigh(S)
    #idx = np.argsort(e_vals)[::-1]
    #e_vecs = e_vecs[:,idx]
    #e_vals = e_vals[idx]

    # each column is an eigenvector



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



def get_ba_cov_matrix(n_genes, cov, p=None, m=2, get_node_edge_sum=False):#,prop=False,, rho=None  cov2 = None,rho2=None):
    '''Based off of the Gershgorin circle theorem, we can expect
    that the code will eventually produce mostly matrices
    that aren't positive definite as the covariance value
    increases and/or more edges added to incidence matrix'''
    while True:
        #if rho == None:
        if p == None:
            ntwk = nx.barabasi_albert_graph(n_genes, m)

        else:
            #ntwk = random_walks_powerlaw_cluster_graph(n=n_genes,m=m,cc=cc,seed=None)
            # p = Probability of adding a triangle after adding a random edge
            ntwk = nx.powerlaw_cluster_graph(n=n_genes,m=m,p=p)


        ntwk_np = nx.to_numpy_matrix(ntwk)
        #else:
        #    ntwk_np, rho_estimate = get_correlated_rndm_ntwrk(n_genes, m=m, rho=rho, count_threshold = 10000)

        #if cov2 == None:
        C = ntwk_np * cov
        #else:
        #    C = ntwk_np * cov
        #    C2 = ntwk_np * cov2
        #    np.fill_diagonal(C2, 1)

        np.fill_diagonal(C, 1)

        #if prop == True and cov2 == None:
        #    diag_C = np.tril(C, k =-1)
        #    i,j = np.nonzero(diag_C)
        #    ix = np.random.choice(len(i), int(np.floor((1-prop) * len(i))), replace=False)
        #    C[np.concatenate((i[ix],j[ix]), axis=None), np.concatenate((j[ix],i[ix]), axis=None)] = -1*cov

        #if cov2 == None:
        if np.all(np.linalg.eigvals(C) > 0) == True:
            if p==None:
                return C
            else:
                return C, nx.average_clustering(ntwk)
            #else:
            #    return C, rho_estimate

        #else:
        #    return C
        #    if (np.all(np.linalg.eigvals(C) > 0) == True) and (np.all(np.linalg.eigvals(C2) > 0) == True):
        #        return C, C2






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


def get_count_pop(lambdas, cov):
    mult_norm = np.random.multivariate_normal(np.asarray([0]* len(lambdas)), cov)#, tol=1e-6)
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
    # don't count p5

    return ['m5','m6','p1','p2','p4', 'p5']


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


def hellinger_transform(array):
    return np.sqrt((array.T/array.sum(axis=1)).T )
    #df = pd.read_csv(mydir + 'data/Tenaillon_et_al/gene_by_pop_delta.txt', sep = '\t', header = 'infer', index_col = 0)
    #return(df.div(df.sum(axis=1), axis=0).applymap(np.sqrt))


def get_random_matrix(c_in):
    #```GNU Lesser General Public License v3.0 code from https://github.com/maclandrol/FisherExact```
    # f2py -c -m asa159 asa159.f90
    #c = array
    # remove empty columns
    empty_cols = (np.where(~c_in.any(axis=0))[0])
    empty_rows = (np.where(~c_in.any(axis=1))[0])
    c = np.delete(c_in, empty_cols, axis=1)
    c = np.delete(c, empty_rows, axis=0)

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

        #for empty_column in empty_c:
        #    np.insert(tmp_observed, empty_column, nr, axis=1)
        rndm_matrix = np.reshape(tmp_observed, (nr,nc))
        for empty_column in empty_cols:
            rndm_matrix = np.insert(rndm_matrix, empty_column, 0, axis=1)
        for empty_row in empty_rows:
            rndm_matrix = np.insert(rndm_matrix, empty_row, 0, axis=0)

        return rndm_matrix
        #return np.reshape(tmp_observed, (nr,nc))



def get_L_stat(e_max_value, n, p):
    mu = (np.sqrt(n-1) + np.sqrt(p)) ** 2
    sigma = (np.sqrt(n-1) + np.sqrt(p)) * (((1/np.sqrt(n-1)) + (1/np.sqrt(p))) ** (1/3))
    return (e_max_value - mu) / sigma



def get_x_stat(e_values, n_features=None):

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

    if n_features == None:
        n = get_n_prime(e_values)
    else:
        n = n_features

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



def get_F_2(PC_space, N_list):
    '''
    Modified F-statistic from Anderson et al., 2017 doi: 10.1111/anzs.12176
    Function assumes that the rows of the count matrix are sorted by group
    i.e., group one is first N1 rows, group two is N2, etc
    '''
    #N = N1+N2
    N = sum(N_list)
    dist_matrix = euclidean_distances(PC_space, PC_space)
    A = -(1/2) * (dist_matrix ** 2)
    I = np.identity(N)
    J_N = np.full((N, N), 1)
    G = (I - ((1/N) * J_N )) @ A @ (I - ((1/N) * J_N ))
    # n matrix list
    n_list = []
    for N_i in N_list:
        n_list.append((1/N_i) * np.full((N_i, N_i), 1))
    #n1 = (1/N1) * np.full((N1, N1), 1)
    #n2 = (1/N2) * np.full((N2, N2), 1)
    #H = block_diag(n1, n2) - ((1/N) * J_N )
    H = block_diag(*n_list) - ((1/N) * J_N )
    # indicator matrices
    # get V matrices
    V_list = []
    for i in range(len(N_list)):
        if i == 0:
            U_i = np.diag( N_list[i]*[1] + sum(N_list[i+1:])*[0])
        elif i == len(N_list) - 1:
            U_i = np.diag( sum(N_list[:i])*[0] + N_list[i]*[1] )
        else:
            U_i = np.diag( sum(N_list[:i])*[0] + N_list[i]*[1] +  sum(N_list[i+1:])*[0])

        V_i = np.trace(((I - H) @ U_i @ (I - H)) @ G ) / (N_list[i]-1)
        V_list.append(V_i)

    F_2 = np.trace(H @ G) / sum( [ (1 - (N_list[i]/N) ) *  V_list[i] for i in range(len(N_list)) ]  )



    return F_2

    #U_i = np.diag( (N1*[1]) + (N2*[0]))

    #U_1 = np.diag( (N1*[1]) + (N2*[0]))
    #U_2 = np.diag( (N1*[0]) + (N2*[1]))

    #V_1 = np.trace(((I - H) @ U_1 @ (I - H)) @ G ) / (N1-1)
    #V_2 = np.trace(((I - H) @ U_2 @ (I - H)) @ G ) / (N2-1)

    #F_2 = np.trace(H @ G) / (((1- (N1/N)) * V_1) + ((1- (N2/N)) * V_2))

    #return F_2, V_1, V_2



def get_F_1(PC_space, N_list):
    g = len(N_list) - 1
    N = sum(N_list)
    dist_matrix = euclidean_distances(PC_space, PC_space)
    A = -(1/2) * (dist_matrix ** 2)
    I = np.identity(N)
    J_N = np.full((N, N), 1)
    G = (I - ((1/N) * J_N )) @ A @ (I - ((1/N) * J_N ))
    # n matrix list
    n_list = []
    for N_i in N_list:
        n_list.append((1/N_i) * np.full((N_i, N_i), 1))
    #n1 = (1/N1) * np.full((N1, N1), 1)
    #n2 = (1/N2) * np.full((N2, N2), 1)
    #H = block_diag(n1, n2) - ((1/N) * J_N )
    H = block_diag(*n_list) - ((1/N) * J_N )

    F = (np.trace(H @ G) / (g-1)) / (np.trace((I - H)@G) / (N-g))

    return F


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
