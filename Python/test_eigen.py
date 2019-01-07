from __future__ import division
import os, re
from collections import Counter
import numpy as np
import pandas as pd
import parevol_tools as pt
from sklearn.decomposition import PCA


df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
df_delta = pt.likelihood_matrix(df, 'Tenaillon_et_al').get_likelihood_matrix()
X = pt.hellinger_transform(df_delta)
pca = PCA()
df_out = pca.fit_transform(X)

#X_centered = X - np.mean(X, axis=0)
#cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
#eigenvalues = pca.explained_variance_
#for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):
#    print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
#    print(eigenvalue)

def get_n_prime(e_values):
    # moments estimator from Patterson et al 2006
    m = len(e_values) + 1
    sq_sum_ev = sum(e_values) ** 2
    sum_sq_ev = sum( e **2 for e in  e_values )
    return ((m+1) * sq_sum_ev) /  (( (m-1)  * sum_sq_ev ) -  sq_sum_ev )



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



e_values = pca.explained_variance_[:-1]

print(get_x_stat(e_values))
#print( get_n_prime(e_values) )
