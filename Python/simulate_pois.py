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
    print(row_list)
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
    #C = get_line_cov_matrix(len(lambdas), cov, alternate_cov_sign=False)
    C =cov
    mult_norm = np.random.multivariate_normal(np.asarray([0]* len(lambdas)), C)#, tol=1e-6)
    mult_norm_cdf = stats.norm.cdf(mult_norm)
    counts = [ get_pois_sample(lambdas[i], mult_norm_cdf[i]) for i in range(len(lambdas))  ]

    return np.asarray(counts)


def get_block_cov(n_genes, var=1, pos_cov = 0.9, neg_cov= -0.9):
    cov = np.full((n_genes, n_genes), float(0))
    cutoff = int(n_genes/2)
    for i in range(n_genes):
        cov[i,i] = var
        for j in range(n_genes):
            if i < j:
                if ((i < cutoff) and ( j < cutoff)) or ((i >= cutoff) and ( j >= cutoff)):
                    cov[i,j] = pos_cov
                    cov[j,i] = pos_cov
                else:
                    cov[i,j] = neg_cov
                    cov[j,i] = neg_cov
    return cov


def run_block_cov_sims():
    df_out=open(pt.get_path() + '/data/simulations/cov_block_euc.txt', 'w')
    n_pops=20
    n_genes=50
    lambda_genes = np.random.gamma(shape=3, scale=1, size=n_genes)
    df_out.write('\t'.join(['Cov', 'Iteration', 'z_score']) + '\n')
    covs = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
    for cov in covs:
        C = get_block_cov(n_genes, pos_cov = cov, neg_cov = -1*cov)
        for i in range(1000):
            print(str(cov) , ' ', str(i))
            test_cov = np.stack( [get_count_pop(lambda_genes, cov= C) for x in range(n_pops)] , axis=0 )
            X = pt.hellinger_transform(test_cov)
            pca = PCA()
            pca_fit = pca.fit_transform(X)
            euc_dist = pt.get_mean_pairwise_euc_distance(pca_fit)
            sim_eucs = []
            for j in range(1000):
                X_j = pt.hellinger_transform(pt.random_matrix(test_cov))
                pca_fit_j = pca.fit_transform(X_j)
                sim_eucs.append( pt.get_mean_pairwise_euc_distance(pca_fit_j) )
            z_score = (euc_dist - np.mean(sim_eucs)) / np.std(sim_eucs)
            df_out.write('\t'.join([str(cov), str(i), str(z_score)]) + '\n')

    df_out.close()


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



def get_fig():
    df = pd.read_csv(pt.get_path() + '/data/simulations/cov_block_euc.txt', sep='\t')
    x = df.Cov.values
    y = df.z_score.values

    fig = plt.figure()
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    x_slope = np.linspace(0,1,1000)
    y_slope = intercept + ( slope *  x_slope)

    plt.scatter(x, y, c='#175ac6', marker = 'o', s = 70, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.2, zorder=2)#, edgecolors='none')
    plt.plot(x_slope, y_slope, c='k', lw =2)
    plt.axhline(y=0, color='red', lw=2, linestyle='--')
    plt.xlabel('Covariance')
    plt.ylabel('Z-score')

    plt.tight_layout()
    fig_name = pt.get_path() + '/figs/z_score_block_regress.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


#run_block_cov_sims()
get_fig()
