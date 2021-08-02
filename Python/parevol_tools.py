from __future__ import division
import os, pickle, math, random, itertools, re
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

#import networkx as nx
from asa159 import rcont2
from copy import copy
import matplotlib.colors as cls

from Bio import SeqIO


np.random.seed(123456789)
random.seed(123456789)



bases_to_skip = ['K', 'S', 'R', 'N', 'Y', 'M', 'W']
base_table = {'A':'T','T':'A','G':'C','C':'G',
            'Y':'R', 'R':'Y', 'S':'W', 'W':'S', 'K':'M', 'M':'K', 'N':'N'}

codon_table = { 'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'CGT': 'R', 'CGC': 'R', 'CGA':'R',
'CGG':'R', 'AGA':'R', 'AGG':'R', 'AAT':'N', 'AAC':'N', 'GAT':'D', 'GAC':'D', 'TGT':'C', 'TGC':'D',
'CAA':'Q', 'CAG':'Q', 'GAA':'E', 'GAG':'E', 'GGT':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G', 'CAT':'H',
'CAC':'H', 'ATT':'I', 'ATC':'I', 'ATA':'I', 'TTA':'L', 'TTG':'L', 'CTT':'L', 'CTC':'L', 'CTA':'L',
'CTG':'L', 'AAA':'K', 'AAG':'K', 'ATG':'M', 'TTT':'F', 'TTC':'F', 'CCT':'P', 'CCC':'P', 'CCA':'P',
'CCG':'P', 'TCT':'S', 'TCC':'S', 'TCA':'S', 'TCG':'S', 'AGT':'S', 'AGC':'S', 'ACT':'T', 'ACC':'T',
'ACA':'T', 'ACG':'T', 'TGG':'W', 'TAT':'Y', 'TAC':'Y', 'GTT':'V', 'GTC':'V', 'GTA':'V', 'GTG':'V',
'TAA':'!', 'TGA':'!', 'TAG':'!'}#, 'KTC':'F', 'KAC':'Y', 'KCC':'A', 'KGC':'D'}



# calculate number of synonymous opportunities for each codon
codon_synonymous_opportunity_table = {}
for codon in codon_table.keys():
    codon_synonymous_opportunity_table[codon] = {}
    for i in range(0,3):
        codon_synonymous_opportunity_table[codon][i] = -1 # since G->G is by definition synonymous, but we don't want to count it
        codon_list = list(codon)
        for base in ['A','C','T','G']:
            codon_list[i]=base
            new_codon = "".join(codon_list)
            if 'K' in new_codon:
                continue
            if codon_table[codon]==codon_table[new_codon]:
                # synonymous!
                codon_synonymous_opportunity_table[codon][i]+=1

bases = set(['A','C','T','G'])
substitutions = []
for b1 in bases:
    for b2 in bases:
        if b2==b1:
            continue

        substitutions.append( '%s->%s' % (b1,b2) )

codon_synonymous_substitution_table = {}
codon_nonsynonymous_substitution_table = {}
for codon in codon_table.keys():
    codon_synonymous_substitution_table[codon] = [[],[],[]]
    codon_nonsynonymous_substitution_table[codon] = [[],[],[]]

    for i in range(0,3):
        reference_base = codon[i]

        codon_list = list(codon)
        for derived_base in ['A','C','T','G']:
            if derived_base==reference_base:
                continue
            substitution = '%s->%s' % (reference_base, derived_base)
            codon_list[i]=derived_base
            new_codon = "".join(codon_list)
            if codon_table[codon]==codon_table[new_codon]:
                # synonymous!
                codon_synonymous_substitution_table[codon][i].append(substitution)
            else:
                codon_nonsynonymous_substitution_table[codon][i].append(substitution)






def calculate_synonymous_nonsynonymous_target_sizes(taxon):
    position_gene_map, effective_gene_lengths, substitution_specific_synonymous_fraction  = create_annotation_map(taxon=taxon)
    return effective_gene_lengths['synonymous'], effective_gene_lengths['nonsynonymous'], substitution_specific_synonymous_fraction


def calculate_reverse_complement_sequence(dna_sequence):
    return "".join(base_table[base] for base in dna_sequence[::-1])


def calculate_codon_sequence(dna_sequence):
    return "".join(codon_table[dna_sequence[3*i:3*i+3]] for i in range(0,len(dna_sequence)/3))




def get_alpha():
    return 0.05


def get_ref_gbff_dict(experiment):

    ref_dict = {"tenaillon": "data/Tenaillon_et_al/sequence.gb"}

    return ref_dict[experiment]


def get_path():
    return os.path.expanduser("~/GitHub/ParEvol")


def get_genome_size(taxon):
    genome_size_dict = {"tenaillon": 4629812}

    return genome_size_dict[taxon]





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






def parse_reference_genome(taxon):
    filename= get_path() + '/' + get_ref_gbff_dict(taxon)

    reference_sequences = []

    # GBK file
    if filename[-3:] == 'gbk':
        file = open(filename,"r")
        origin_reached = False
        for line in file:
            if line.startswith("ORIGIN"):
                origin_reached=True
            if origin_reached:
                items = line.split()
                if items[0].isdigit():
                    reference_sequences.extend(items[1:])
        file.close()

    # FASTA file
    else:
        file = open(filename,"r")
        file.readline() # header
        for line in file:
            reference_sequences.append(line.strip())
        file.close()

    reference_sequence = "".join(reference_sequences).upper()
    return reference_sequence




#####################################################################
#
# Reads through the Genbank file for the reference and
# compiles a list of genes, tRNAs, etc.
#
#####################################################################
def parse_gene_list(taxon, reference_sequence=None):
    gene_names = []
    start_positions = []
    end_positions = []
    promoter_start_positions = []
    promoter_end_positions = []
    gene_sequences = []
    strands = []
    genes = []
    features = []
    protein_ids = []

    filename= get_path() + '/' + get_ref_gbff_dict(taxon)
    gene_features = ['CDS', 'tRNA', 'rRNA', 'ncRNA', 'tmRNA']
    recs = [rec for rec in SeqIO.parse(filename, "genbank")]
    count_riboswitch = 0
    for rec in recs:
        reference_sequence = rec.seq
        contig = rec.annotations['accessions'][0]
        for feat in rec.features:
            if 'pseudo' in list((feat.qualifiers.keys())):
                continue
            if (feat.type == "source") or (feat.type == "gene"):
                continue

            locations = re.findall(r"[\w']+", str(feat.location))
            if feat.type in gene_features:
                locus_tag = feat.qualifiers['locus_tag'][0]
            elif (feat.type=="regulatory"):
                locus_tag = feat.qualifiers["regulatory_class"][0] + '_' + str(count_riboswitch)
                count_riboswitch += 1
            else:
                continue
            # for frameshifts, split each CDS seperately and merge later
            # Fix this for Deinococcus, it has a frameshift in three pieces
            split_list = []
            if 'join' in locations:
                location_str = str(feat.location)
                minus_position = []
                if '-' in location_str:
                    minus_position = [r.start() for r in re.finditer('-', location_str)]
                pos_position = []

                if '+' in location_str:
                    if taxon == 'D':
                        pos_position = [pos for pos, char in enumerate(location_str) if char == '+']
                    elif taxon == 'J':
                        pos_position = [pos for pos, char in enumerate(location_str) if char == '+']
                    else:
                        pos_position = [r.start() for r in re.finditer('+', location_str)]


                if len(minus_position) + len(pos_position) == 2:
                    if len(minus_position) == 2:
                        strand_symbol_one = '-'
                        strand_symbol_two = '-'
                    elif len(pos_position) == 2:
                        strand_symbol_one = '+'
                        strand_symbol_two = '+'
                    else:
                        # I don't think this is possible, but might as well code it up
                        if minus_position[0] < pos_position[0]:
                            strand_symbol_one = '-'
                            strand_symbol_two = '+'
                        else:
                            strand_symbol_one = '+'
                            strand_symbol_two = '-'

                    start_one = int(locations[1])
                    stop_one = int(locations[2])
                    start_two = int(locations[3])
                    stop_two = int(locations[4])

                    locus_tag1 = locus_tag + '_1'
                    locus_tag2 = locus_tag + '_2'

                    split_list.append([locus_tag1, start_one, stop_one, strand_symbol_one])
                    split_list.append([locus_tag2, start_two, stop_two, strand_symbol_two])

                else:
                    if len(pos_position) == 3:
                        strand_symbol_one = '+'
                        strand_symbol_two = '+'
                        strand_symbol_three = '+'
                    start_one = int(locations[1])
                    stop_one = int(locations[2])
                    start_two = int(locations[3])
                    stop_two = int(locations[4])
                    start_three = int(locations[5])
                    stop_three = int(locations[6])

                    locus_tag1 = locus_tag + '_1'
                    locus_tag2 = locus_tag + '_2'
                    locus_tag3 = locus_tag + '_3'

                    split_list.append([locus_tag1, start_one, stop_one, strand_symbol_one])
                    split_list.append([locus_tag2, start_two, stop_two, strand_symbol_two])
                    split_list.append([locus_tag3, start_three, stop_three, strand_symbol_three])


            else:
                strand_symbol = str(feat.location)[-2]
                start = int(locations[0])
                stop = int(locations[1])
                split_list.append([locus_tag, start, stop, strand_symbol])

            for split_item in split_list:
                locus_tag = split_item[0]
                start = split_item[1]
                stop = split_item[2]
                strand_symbol = split_item[3]


                if feat.type == 'CDS':
                    #  why was a -1 there originally?
                    #gene_sequence = reference_sequence[start-1:stop]
                    gene_sequence = str(reference_sequence[start:stop])
                else:
                    gene_sequence = ""


                if 'gene' in list((feat.qualifiers.keys())):
                    gene = feat.qualifiers['gene'][0]
                else:
                    gene = ""

                if 'protein_id' in list((feat.qualifiers.keys())):
                    protein_id = feat.qualifiers['protein_id'][0]
                else:
                    protein_id = ""


                if strand_symbol == '+':
                    promoter_start = start - 100 # by arbitrary definition, we treat the 100bp upstream as promoters
                    promoter_end = start - 1
                    strand = 'forward'
                else:
                    promoter_start = stop+1
                    promoter_end = stop+100
                    strand = 'reverse'


                if gene_sequence!="" and (not len(gene_sequence)%3==0):
                    print(locus_tag, start, "Not a multiple of 3")
                    continue

                # dont need to check if gene names are unique because we're using
                # locus tags

                start_positions.append(start)
                end_positions.append(stop)
                promoter_start_positions.append(promoter_start)
                promoter_end_positions.append(promoter_end)
                gene_names.append(locus_tag)
                gene_sequences.append(gene_sequence)
                strands.append(strand)
                genes.append(gene)
                features.append(feat.type)
                protein_ids.append(protein_id)

    gene_names, start_positions, end_positions, promoter_start_positions, promoter_end_positions, gene_sequences, strands, genes, features, protein_ids = (list(x) for x in zip(*sorted(zip(gene_names, start_positions, end_positions, promoter_start_positions, promoter_end_positions, gene_sequences, strands, genes, features, protein_ids), key=lambda pair: pair[1])))

    return gene_names, np.array(start_positions), np.array(end_positions), np.array(promoter_start_positions), np.array(promoter_end_positions), gene_sequences, strands, genes, features, protein_ids







def create_annotation_map(taxon, gene_data=None):

    if gene_data==None:
        gene_data = parse_gene_list(taxon)

    gene_names, gene_start_positions, gene_end_positions, promoter_start_positions, promoter_end_positions, gene_sequences, strands, genes, features, protein_ids = gene_data
    position_gene_map = {}
    gene_position_map = {}
    # new
    gene_feature_map = {}

    # then greedily annotate genes at remaining sites
    for gene_name, feature, start, end in zip(gene_names, features, gene_start_positions, gene_end_positions):
        gene_feature_map[gene_name] = feature
        for position in range(start,end+1):
            if position not in position_gene_map:
                position_gene_map[position] = gene_name
                if gene_name not in gene_position_map:
                    gene_position_map[gene_name]=[]
                gene_position_map[gene_name].append(position)


    # remove 'partial' genes that have < 10bp unmasked sites
    for gene_name in list(sorted(gene_position_map.keys())):
        if len(gene_position_map[gene_name]) < 10:
            for position in gene_position_map[gene_name]:
                position_gene_map[position] = 'repeat'
            del gene_position_map[gene_name]

    # count up number of synonymous opportunities
    effective_gene_synonymous_sites = {}
    effective_gene_nonsynonymous_sites = {}

    substitution_specific_synonymous_sites = {substitution: 0 for substitution in substitutions}
    substitution_specific_nonsynonymous_sites = {substitution: 0 for substitution in substitutions}

    for gene_name, start, end, gene_sequence, strand in zip(gene_names, gene_start_positions, gene_end_positions, gene_sequences, strands):

        if gene_name not in gene_position_map:
            continue

        if strand=='forward':
            oriented_gene_sequence = gene_sequence
        else:
            oriented_gene_sequence = calculate_reverse_complement_sequence(gene_sequence)

        for position in gene_position_map[gene_name]:

            if gene_name not in effective_gene_synonymous_sites:
                effective_gene_synonymous_sites[gene_name]=0
                effective_gene_nonsynonymous_sites[gene_name]=0

            if 'CDS' not in gene_feature_map[gene_name]:
                continue

            else:
                # calculate position in gene
                if strand=='forward':
                    position_in_gene = position-start
                else:
                    position_in_gene = end-position

                # calculate codon start
                codon_start = int(position_in_gene/3)*3
                if codon_start+3 > len(gene_sequence):
                    continue

                #codon = gene_sequence[codon_start:codon_start+3]
                codon = oriented_gene_sequence[codon_start:codon_start+3]
                if any(codon_i in codon for codon_i in bases_to_skip):
                    continue
                position_in_codon = position_in_gene%3


                effective_gene_synonymous_sites[gene_name] += codon_synonymous_opportunity_table[codon][position_in_codon]/3.0
                effective_gene_nonsynonymous_sites[gene_name] += 1-codon_synonymous_opportunity_table[codon][position_in_codon]/3.0

                for substitution in codon_synonymous_substitution_table[codon][position_in_codon]:
                    substitution_specific_synonymous_sites[substitution] += 1

                for substitution in codon_nonsynonymous_substitution_table[codon][position_in_codon]:
                    substitution_specific_nonsynonymous_sites[substitution] += 1



    substitution_specific_synonymous_fraction = {substitution: substitution_specific_synonymous_sites[substitution]*1.0/(substitution_specific_synonymous_sites[substitution]+substitution_specific_nonsynonymous_sites[substitution]) for substitution in substitution_specific_synonymous_sites.keys()}
    # then annotate promoter regions at remaining sites
    for gene_name,start,end in zip(gene_names,promoter_start_positions,promoter_end_positions):
        for position in range(start,end+1):
            if position not in position_gene_map:
                # position hasn't been annotated yet

                if gene_name not in gene_position_map:
                    # the gene itself has not been annotated
                    # so don't annotate the promoter
                    continue
                else:
                    position_gene_map[position] = gene_name
                    gene_position_map[gene_name].append(position)

    # calculate effective gene lengths
    effective_gene_lengths = {gene_name: len(gene_position_map[gene_name])-effective_gene_synonymous_sites[gene_name] for gene_name in gene_position_map.keys()}
    effective_gene_lengths['synonymous'] = sum([effective_gene_synonymous_sites[gene_name] for gene_name in gene_position_map.keys()])
    effective_gene_lengths['nonsynonymous'] = sum([effective_gene_nonsynonymous_sites[gene_name] for gene_name in gene_position_map.keys()])
    effective_gene_lengths['noncoding'] = get_genome_size(taxon=taxon)-effective_gene_lengths['synonymous']-effective_gene_lengths['nonsynonymous']


    return position_gene_map, effective_gene_lengths, effective_gene_synonymous_sites, substitution_specific_synonymous_fraction
