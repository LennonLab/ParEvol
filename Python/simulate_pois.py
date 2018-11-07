from __future__ import division
import math, random, itertools
import numpy as np
import pandas as pd
import scipy.stats as stats
import parevol_tools as pt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_ring_cov_matrix(n_rows, cov, var = 1):
    row_list = []
    for i in range(n_rows):
        list_i = [0] * n_rows
        list_i[i] = var
        if i == 0:
            list_i[i+1] = cov
            list_i[n_rows-1] = cov
        elif 0 < i < (n_rows-1):
            list_i[i-1] = cov
            list_i[i+1] = cov
        else:
            list_i[0] = cov
            list_i[i-1] = cov
        row_list.append(list_i)
    return np.asarray(row_list)


def get_line_cov_matrix(n_rows, cov, var=1,alternate_cov_sign=False):
    row_list = []
    cov_sign_dict = {}
    for i in range(n_rows-1):
        #cov = np.random.normal(loc=0.0, scale=5.0)
        if i%2 == 1:
            if alternate_cov_sign == True:
                cov_sign_dict[str(i)+'_'+str(i+1)] = -cov
            else:
                cov_sign_dict[str(i)+'_'+str(i+1)] = cov
        else:
            cov_sign_dict[str(i)+'_'+str(i+1)] = cov

    for i in range(n_rows):
        list_i = [0] * n_rows
        list_i[i] = var
        if i == 0:
            list_i[i+1] = cov_sign_dict[str(i)+'_'+str(i+1)]
        elif 0 < i < (n_rows-1):
            list_i[i-1] = cov_sign_dict[str(i-1)+'_'+str(i)]
            list_i[i+1] = cov_sign_dict[str(i)+'_'+str(i+1)]
        else:
            list_i[i-1] = cov_sign_dict[str(i-1)+'_'+str(i)]
        row_list.append(list_i)
    return np.asarray(row_list)


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
    C = get_line_cov_matrix(len(lambdas), cov, alternate_cov_sign=False)
    #print(C)
    mult_norm = np.random.multivariate_normal(np.asarray([0]* len(lambdas)), C)#, tol=1e-6)
    mult_norm_cdf = stats.norm.cdf(mult_norm)
    counts = [ get_pois_sample(lambdas[i], mult_norm_cdf[i]) for i in range(len(lambdas))  ]

    return np.asarray(counts)


def run_all_sims():
    df_out=open(pt.get_path() + '/data/simulations/cov_euc.txt', 'w')
    n_pops=20
    n_genes=50
    lambda_genes = np.random.gamma(shape=3, scale=1, size=n_genes)
    covs = [0.5, 0, -0.5]
    df_out.write('\t'.join(['Covariance', 'Iteration', 'z_score']) + '\n')
    for cov in covs:
        for i in range(1000):
            print(str(cov) + ' '  + str(i))
            test_cov = np.stack( [get_count_pop(lambda_genes, cov= cov) for x in range(n_pops)] , axis=0 )
            X = pt.hellinger_transform(test_cov)
            pca = PCA()
            pca_fit = pca.fit_transform(X)
            euc_dist = pt.get_euclidean_distance(pca_fit)
            sim_eucs = []
            for j in range(1000):
                #if j % 100 == 0:
                #    print(j)
                X_j = pt.hellinger_transform(pt.random_matrix(test_cov))
                pca_fit_j = pca.fit_transform(X_j)
                sim_eucs.append( pt.get_euclidean_distance(pca_fit_j) )
            z_score = (euc_dist - np.mean(sim_eucs)) / np.std(sim_eucs)

            df_out.write('\t'.join([str(cov), str(i), str(z_score)]) + '\n')

    df_out.close()


#run_all_sims()
covs = [0.5, 0.1]
lambda_genes_ = np.random.gamma(shape=3, scale=1, size=2)
print(lambda_genes_)
for cov in covs:
    count_diff = []
    for x in range(1000):
        pop_ = get_count_pop(lambda_genes_, cov= -0.5)
        #print(pop_[0], pop_[1])
        count_diff.append(abs( (pop_[0] - pop_[1]) / (pop_[0] + pop_[1]) ))
    print(cov, np.mean(count_diff) )

#print(pop_)

def get_fig():
    df = pd.read_csv(pt.get_path() + '/data/simulations/cov_euc.txt', sep='\t')
    p_5 = df.loc[df['Covariance'] == 0.5].z_score.values
    m_5 = df.loc[df['Covariance'] == -0.5].z_score.values
    _0 = df.loc[df['Covariance'] == 0].z_score.values
    fig = plt.figure()
    plt.hist(p_5,bins=50, alpha=0.4, color = 'blue')
    plt.hist(m_5,bins=50, alpha=0.4, color = 'red')
    plt.hist(_0,bins=50, alpha=0.4, color = 'grey')

    plt.xlabel('Z-score')
    plt.ylabel('Frequency')

    plt.tight_layout()
    fig_name = pt.get_path() + '/figs/z_score_hist.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()
