from __future__ import division
import os, pickle, operator
from itertools import compress
import numpy as np
import pandas as pd
import parevol_tools as pt
from collections import Counter
from sklearn.decomposition import PCA
import networkx as nx



def run_ba_ntwk_cov_sims(iter1=1000, iter2=1000):
    df_out=open(pt.get_path() + '/data/simulations/cov_ntwrk_pop.txt', 'w')
    #n_pops=100
    n_genes=50
    ntwk = nx.barabasi_albert_graph(n_genes, 2)
    ntwk_np = nx.to_numpy_matrix(ntwk)
    df_out.write('\t'.join(['N', 'Cov', 'Iteration', 'euc_percent']) + '\n')
    covs = [0.05, 0.1, 0.15, 0.2]
    Ns = [2, 4, 8, 16, 32, 64]
    for N in Ns:
        for cov in covs:
            C = ntwk_np * cov
            np.fill_diagonal(C, 1)
            for i in range(iter1):
                lambda_genes = np.random.gamma(shape=1, scale=1, size=n_genes)
                test_cov = np.stack( [pt.get_count_pop(lambda_genes, cov= C) for x in range(N)] , axis=0 )
                X = pt.hellinger_transform(test_cov)
                pca = PCA()
                pca_fit = pca.fit_transform(X)
                euc_dist = pt.get_mean_pairwise_euc_distance(pca_fit)
                euc_dists = []
                for j in range(iter2):
                    X_j = pt.hellinger_transform(pt.random_matrix(test_cov))
                    pca_fit_j = pca.fit_transform(X_j)
                    euc_dists.append( pt.get_mean_pairwise_euc_distance(pca_fit_j) )
                euc_percent = len( [k for k in euc_dists if k < euc_dist] ) / len(euc_dists)
                print(N, cov, i, euc_percent)
                df_out.write('\t'.join([str(N), str(cov), str(i), str(euc_percent)]) + '\n')

    df_out.close()



def rndm_sample_tenaillon(iter1=1000, iter2=1000):
    df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    n_rows = df_np.shape[0]
    df_out=open(pt.get_path() + '/data/Tenaillon_et_al/sample_size_sim.txt', 'w')
    df_out.write('\t'.join(['N', 'G', 'Iteration', 'euc_percent', 'z_score']) + '\n')
    Ns = [5]#list(range(2, n_rows))
    for N in Ns:
        for i in range(1000):
            #df_np_i = df_np[np.random.choice(n_rows, N, replace=False), :]
            #df_np_i = df_np_i[: , ~np.all(df_np_i == 0, axis=0)]
            #df_i = df.sample(N)
            df_np_i = df_np[np.random.randint(n_rows, size=N), :]
            gene_bool = np.all(df_np_i == 0, axis=0)
            # flip around to select gene_size
            gene_names_i = list(compress(gene_names, list(map(operator.not_, gene_bool))))
            df_np_i = df_np_i[:,~np.all(df_np_i == 0, axis=0)]
            #df_i = df_i.loc[:, (df_i != 0).any(axis=0)]
            np.seterr(divide='ignore')
            df_np_i_delta = pt.likelihood_matrix_array(df_np_i, gene_names_i, 'Tenaillon_et_al').get_likelihood_matrix()
            X = pt.hellinger_transform(df_np_i_delta)
            pca = PCA()
            pca_fit = pca.fit_transform(X)
            euc_dist = pt.get_mean_pairwise_euc_distance(pca_fit)
            euc_dists = []
            for j in range(iter2):
                df_np_i_j = pt.random_matrix(df_np_i)
                np.seterr(divide='ignore')
                df_np_i_j_delta = pt.likelihood_matrix_array(df_np_i_j, gene_names_i, 'Tenaillon_et_al').get_likelihood_matrix()
                #df_i_j = pd.DataFrame(data=pt.random_matrix(df_np_i_j), index=df_i.index, columns=df_i.columns)
                #df_i_j_delta = pt.likelihood_matrix(df_i_j, 'Tenaillon_et_al').get_likelihood_matrix()
                X_j = pt.hellinger_transform(df_np_i_j_delta)
                pca_fit_j = pca.fit_transform(X_j)
                euc_dists.append( pt.get_mean_pairwise_euc_distance(pca_fit_j) )

            G = df_np_i.shape[1]
            euc_percent = len( [k for k in euc_dists if k < euc_dist] ) / len(euc_dists)
            z_score = (euc_dist - np.mean(euc_dists)) / np.std(euc_dists)
            print(str(N), str(G), str(i), str(euc_percent), str(z_score))
            df_out.write('\t'.join([str(N), str(G), str(i), str(euc_percent), str(z_score)]) + '\n')

    df_out.close()



rndm_sample_tenaillon()

# include simulation with the neutral substitution rate relative to the adaptive



def get_pop_matrix(n_pops, n_genes, subs, probs, env):
    n_pops_dict = {}
    for i in range(n_pops):
        mutation_counts = np.random.choice(n_genes, size = subs, replace=True, p = probs)
        mutation_counts_dict = Counter(mutation_counts)

        n_pops_dict[env + '_' + str(i)] = mutation_counts_dict

    return n_pops_dict


#run_ba_ntwk_cov_sims()
