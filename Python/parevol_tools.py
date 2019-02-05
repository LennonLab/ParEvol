from __future__ import division
import os, pickle, math, random, itertools
from itertools import combinations
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import matplotlib.colors as cls
import rpy2.robjects as robjects
import clean_data as cd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import comb
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

#np.random.seed(123456789)


def get_path():
    return os.path.expanduser("~/GitHub/ParEvol")


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
    C =cov
    mult_norm = np.random.multivariate_normal(np.asarray([0]* len(lambdas)), C)#, tol=1e-6)
    mult_norm_cdf = stats.norm.cdf(mult_norm)
    counts = [ get_pois_sample(lambdas[i], mult_norm_cdf[i]) for i in range(len(lambdas))  ]

    return np.asarray(counts)


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


def get_mean_centroid_distance(array, k = 3):
    X = array[:,0:k]
    #centroids = np.mean(X, axis = 0)
    return np.mean(np.sqrt(np.sum(np.square(X - np.mean(X, axis = 0)), axis=1)))

    #for row in X:
    #    centroid_distances.append(np.linalg.norm(row-centroids))
    #return np.mean(centroid_distances)


def get_euc_magnitude_diff(array, k = 3):
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





def get_mean_pairwise_euc_distance(array, k = 3):
    X = array[:,0:k]
    row_sum = np.sum( euclidean_distances(X, X), axis =1)
    return sum(row_sum) / ( len(row_sum) * (len(row_sum) -1)  )


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


def get_x_stat(e_values):

    def get_n_prime(e_values):
        # moments estimator from Patterson et al 2006
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


def number_matrices(array):
    print("what")



def get_theta_from_cov(C):
    eVa, eVe = np.linalg.eig(C)
    var_1 = C[0,0]
    var_2 = C[1,1]
    if C[0,1] > 0.0:
        if abs(round(math.degrees(math.acos(eVe[0,0])), 3)) > 90:
            theta = (180 - abs(round(math.degrees(math.acos(eVe[0,0])), 3)))
        else:
            theta = abs(round(math.degrees(math.acos(eVe[0,0])), 3))

    elif C[0,1] < 0.0:
        if abs(round(math.degrees(math.acos(eVe[0,0])), 3)) > 90:
            theta = -(180 - abs(round(math.degrees(math.acos(eVe[0,0])), 3)))
        else:
            theta = -abs(round(math.degrees(math.acos(eVe[0,0])), 3))
    else:
        theta = 0
    major_axis_length = 2 * math.sqrt(5.991 * eVa[0])
    minor_axis_length = 2 * math.sqrt(5.991 * eVa[1])
    return major_axis_length, minor_axis_length, theta


def ellipse_polyline(ellipses, n=100):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    st = np.sin(t)
    ct = np.cos(t)
    result = []
    for x0, y0, a, b, angle in ellipses:
        angle = np.deg2rad(angle)
        sa = np.sin(angle)
        ca = np.cos(angle)
        p = np.empty((n, 2))
        p[:, 0] = x0 + a * ca * ct - b * sa * st
        p[:, 1] = y0 + a * sa * ct + b * ca * st
        result.append(p)
    return result




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
        self.df = df.copy()
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
                if ('gene_list' in keyword_parameters):
                    return { gene_name: length_dict[gene_name] for gene_name in keyword_parameters['gene_list'] }
                    #for gene_name in keyword_parameters['gene_list']:
                else:
                    return(length_dict)

    def get_likelihood_matrix(self):
        genes = self.df.columns.tolist()
        genes_lengths = self.get_gene_lengths(gene_list = genes)
        L_mean = np.mean(list(genes_lengths.values()))
        L_i = np.asarray(list(genes_lengths.values()))
        N_genes = len(genes)
        m_mean = self.df.sum(axis=1) / N_genes

        for index, row in self.df.iterrows():
            m_mean_j = m_mean[index]
            np.seterr(divide='ignore')
            delta_j = row * np.log((row * (L_mean / L_i)) / m_mean_j)
            self.df.loc[index,:] = delta_j

        df_new = self.df.fillna(0)
        # remove colums with all zeros
        df_new.loc[:, (df_new != 0).any(axis=0)]
        # replace negative values with zero
        df_new[df_new < 0] = 0
        return df_new



class likelihood_matrix_array:
    def __init__(self, array, gene_list, dataset):
        self.array = np.copy(array)
        self.gene_list = gene_list
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
                if ('gene_list' in keyword_parameters):
                    return { gene_name: length_dict[gene_name] for gene_name in keyword_parameters['gene_list'] }
                else:
                    return(length_dict)

    def get_likelihood_matrix(self):
        genes_lengths = self.get_gene_lengths(gene_list = self.gene_list)
        L_mean = np.mean(list(genes_lengths.values()))
        L_i = np.asarray(list(genes_lengths.values()))
        N_genes = len(self.gene_list)
        #m_mean = self.df.sum(axis=1) / N_genes
        m_mean = np.sum(self.array, axis=0) / N_genes

        #for index, row in self.df.iterrows():
        #    m_mean_j = m_mean[index]
        #    np.seterr(divide='ignore')
        #    delta_j = row * np.log((row * (L_mean / L_i)) / m_mean_j)
        #    self.df.loc[index,:] = delta_j

        # just use matrix operations
        np.seterr(divide='ignore')
        df_new = self.array * np.log((self.array * (L_mean / L_i)) / m_mean)
        np.seterr(divide='ignore')
        #df_new = self.df_new.fillna(0)
        df_new[np.isnan(df_new)] = 0
        # remove colums with all zeros
        #df_new.loc[:, (df_new != 0).any(axis=0)]
        df_new = df_new[:,~np.all(df_new == 0, axis=0)]

        # replace negative values with zero
        #if (df_new<0).any() ==True:
        #    print('Negative #!!!')

        df_new[df_new < 0] = 0
        return df_new


def get_kde(array):
    grid_ = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.1, 10, 50)},
                    cv=20) # 20-fold cross-validation
    grid_.fit(array[:, None])
    x_grid_ = np.linspace(0, 2.5, 1000)
    kde_ = grid_.best_estimator_
    pdf_ = np.exp(kde_.score_samples(x_grid_[:, None]))
    pdf_ = [x / sum(pdf_) for x in pdf_]

    return [x_grid_, pdf_, kde_.bandwidth]



# function to generate confidence intervals based on Fisher Information criteria
def CI_FIC(results):
    # standard errors = square root of the diagnol of a variance-covariance matrix
    ses = np.sqrt(np.absolute(np.diagonal(results.cov_params())))
    cfs = results.params
    lw = cfs - (1.96*ses)
    up = cfs +(1.96*ses)
    return (lw, up)
