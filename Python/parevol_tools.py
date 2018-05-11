from __future__ import division
import os, pickle
#import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import rpy2.robjects as robjects



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
            #print(i, j, BC_ij)
            distance_array[i,j] = BC_ij
            distance_array[j,i] = BC_ij
    return distance_array



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
    centroid_distances = []
    centroids = np.mean(X, axis = 0)
    for row in X:
        centroid_distances.append(np.linalg.norm(row-centroids))
    return np.mean(centroid_distances)



def hellinger_transform(array):
    return np.sqrt((e.T/e.sum(axis=1)).T )
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
            conv_dict = self.parse_convergence_matrix(get_path() + "/data/Good_et_al/gene_convergence_matrix.txt")
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
