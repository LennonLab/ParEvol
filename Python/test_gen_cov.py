from __future__ import division
import os
import numpy as np
import pandas as pd
import parevol_tools as pt
import clean_data as cd
import matplotlib.pyplot as plt
from operator import itemgetter

def plot_tenaillon_gene_pca():
    df_path = os.path.expanduser("~/GitHub/ParEvol/") + 'data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    X = cd.likelihood_matrix_array(df, df.columns.values, 'Tenaillon_et_al').get_likelihood_matrix()
    X = (X > 0).astype(int)
    X = X[:, (X != 0).sum(axis=0) > 1]
    n = X.shape[0]
    p = X.shape[1]
    # covariance matrix
    C = np.identity(n) - (1/n) * np.full((n,n), 1)
    X_C = np.dot(C, X)
    S = np.dot(X_C.T, X_C) / (n-1)
    # Eigen decomposition
    e_vals, e_vecs = np.linalg.eig(S)
    e_vals = np.sort(e_vals)[::-1]

    # correlation matrix
    D = np.diag(np.sqrt(np.diag(S)))

    X_S = np.linalg.multi_dot([C, X, np.linalg.inv(D)])

    R = (1/(n-1)) * np.dot(X_S.T, X_S)
    e_vals, e_vecs = np.linalg.eig(R)
    e_val_vec = list(zip(e_vals, e_vecs))

    e_val_vec.sort(key=itemgetter(0))
    e_val_vec = e_val_vec[::-1]
    e_val_sortd = [x[0] for x in e_val_vec]
    e_val_sortd_rel = [x/sum(e_val_sortd) *100 for x in e_val_sortd]

    broken_stick = []
    for i in range(1, len(e_val_sortd_rel) +1):
        broken_stick.append(   (sum(1 / np.arange(i, len(e_val_sortd_rel) +1)) / len(e_val_sortd_rel)) * 100   )

    fig = plt.figure()
    plt.plot(list(range(1, len(e_val_sortd_rel)+1)), e_val_sortd_rel, linestyle='--', marker='o', color='royalblue', label='Observed')
    plt.plot(list(range(1, len(e_val_sortd_rel)+1)), broken_stick, linestyle=':', alpha=0.7, color='red', label='Broken-stick')

    plt.legend(loc='upper right', fontsize='x-large')

    plt.xlabel('Eigenvalue rank', fontsize = 20)
    plt.ylabel(r'$ \%$' + ' variance explained', fontsize = 20)
    plt.tight_layout()
    fig_name = os.path.expanduser("~/GitHub/ParEvol/") + '/figs/gene_eval_rac.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()






plot_tenaillon_gene_pca()
