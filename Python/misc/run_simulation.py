from __future__ import division
import os, pickle, operator
from itertools import compress
import numpy as np
import pandas as pd
import parevol_tools as pt
from collections import Counter
from sklearn.decomposition import PCA


# z = 1.645 for one sided test with alpha=0.05
def run_ba_cov_sims(gene_list, pop_list, out_name, iter1=1000, iter2=1000):
    df_out=open(pt.get_path() + '/data/simulations/' + out_name + '.txt', 'w')
    df_out.write('\t'.join(['N', 'G', 'Cov', 'Iteration', 'dist_percent']) + '\n')
    covs = [0.1, 0.15, 0.2]
    for G in gene_list:
        for N in pop_list:
            for cov in covs:
                for i in range(iter1):
                    C = pt.get_ba_cov_matrix(G, cov)
                    while True:
                        lambda_genes = np.random.gamma(shape=1, scale=1, size=G)
                        test_cov = np.stack( [pt.get_count_pop(lambda_genes, cov= C) for x in range(N)] , axis=0 )
                        #test_cov_row_sum = test_cov.sum(axis=1)
                        if (np.any(test_cov.sum(axis=1) == 0 )) == False:
                            break
                        #if np.count_nonzero(test_cov_row_sum) == len(test_cov_row_sum):
                        #    break
                    X = pt.hellinger_transform(test_cov)
                    pca = PCA()
                    pca_fit = pca.fit_transform(X)
                    euc_dist = pt.get_mean_pairwise_euc_distance(pca_fit)
                    euc_dists = []
                    for j in range(iter2):
                        X_j = pt.hellinger_transform(pt.get_random_matrix(test_cov))
                        #X_j = pt.hellinger_transform(pt.random_matrix(test_cov))
                        pca_fit_j = pca.fit_transform(X_j)
                        euc_dists.append( pt.get_mean_pairwise_euc_distance(pca_fit_j) )
                    euc_percent = len( [k for k in euc_dists if k < euc_dist] ) / len(euc_dists)
                    print(N, G, cov, i, euc_percent)
                    df_out.write('\t'.join([str(N), str(G), str(cov), str(i), str(euc_percent)]) + '\n')
    df_out.close()


def run_ba_cov_neutral_sims(shape=1, scale=1, G = 50, N = 50, iter1=1000, iter2=1000):
    df_out=open(pt.get_path() + '/data/simulations/ba_cov_neutral_sims.txt', 'w')
    df_out.write('\t'.join(['N', 'G', 'lamba_mean', 'lambda_neutral', 'Cov', 'Iteration', 'dist_percent']) + '\n')
    covs = [0.2]
    mean_gamma = shape * scale
    neutral_range = np.logspace(-2, 1, num=20, endpoint=True, base=10.0)
    neutral_range = neutral_range[::-1]
    for neutral_ in neutral_range:
        for cov in covs:
            for i in range(iter1):
                C = pt.get_ba_cov_matrix(G, cov)
                lambda_genes = np.random.gamma(shape=shape, scale=scale, size=G)
                lambda_genes_null = np.asarray([neutral_] * G)
                test_cov_adapt = np.stack( [pt.get_count_pop(lambda_genes, C= C) for x in range(N)] , axis=0 )
                # matrix with diaganol values equal to one
                test_cov_neutral = np.stack( [pt.get_count_pop(lambda_genes_null, C= np.identity(G)) for x in range(N)] , axis=0 )
                test_cov = test_cov_adapt + test_cov_neutral

                X = pt.hellinger_transform(test_cov)
                pca = PCA()
                pca_fit = pca.fit_transform(X)
                euc_dist = pt.get_mean_pairwise_euc_distance(pca_fit)
                euc_dists = []
                for j in range(iter2):
                    #X_j = pt.hellinger_transform(pt.random_matrix(test_cov))
                    X_j = pt.hellinger_transform(pt.get_random_matrix(test_cov))
                    pca_fit_j = pca.fit_transform(X_j)
                    euc_dists.append( pt.get_mean_pairwise_euc_distance(pca_fit_j) )
                euc_percent = len( [k for k in euc_dists if k < euc_dist] ) / len(euc_dists)
                print(neutral_, cov, i, euc_percent)
                df_out.write('\t'.join([str(N), str(G), str(mean_gamma), str(neutral_), str(cov), str(i), str(euc_percent)]) + '\n')
    df_out.close()


def run_ba_cor_sub_sims(shape=1, scale=1, G = 50, N = 50):
    cov = 0.1
    lambda_genes = np.random.gamma(shape=shape, scale=scale, size=G)
    lambda_genes.sort()
    C, edge_count = pt.get_ba_cov_matrix(G, cov, get_node_edge_sum=True)
    zipped = list(zip( list(range(G)) , edge_count.tolist()[0] ))
    zipped.sort(key = lambda t: t[1])
    # figure out how to sort covariance matrix
    total_inversions = 30
    inversion_count = 0
    while inversion_count < total_inversions:
        pair = np.random.choice(list(range(G)), size = 2, replace=False)
        pair.sort()
        if lambda_genes[pair[0]] < lambda_genes[pair[1]]:
            lambda_0 =lambda_genes[pair[0]].copy()
            lambda_1 =lambda_genes[pair[1]].copy()
            lambda_genes[pair[0]] = lambda_1
            lambda_genes[pair[1]] = lambda_0
            inversion_count += 1
    unzipped = list(zip(*zipped))
    rezipped = list( zip( unzipped[0], unzipped[1],  lambda_genes) )
    rezipped.sort(key = lambda t: t[0])
    unrezipped = list(zip(*rezipped))
    lambda_genes_sorted = list(unrezipped[2])

    print(lambda_genes_sorted)



    #list[min(lambda_genes[pair[0]],j)] < list[max(i,j)]
    #print()

def rndm_sample_tenaillon(iter1=1000, iter2=1000):
    df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    n_rows = df_np.shape[0]
    df_out=open(pt.get_path() + '/data/Tenaillon_et_al/sample_size_sim.txt', 'w')
    df_out.write('\t'.join(['N', 'G', 'Iteration', 'dist_percent', 'z_score']) + '\n')
    Ns = list(range(2, 40, 2))
    for N in Ns:
        for i in range(iter1):
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
                #df_np_i_j = pt.random_matrix(df_np_i)
                df_np_i_j = pt.get_random_matrix(df_np_i)
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
            print(str(N), str(i), str(G), str(euc_percent), str(z_score))
            df_out.write('\t'.join([str(N), str(G), str(i), str(euc_percent), str(z_score)]) + '\n')

    df_out.close()




#run_ba_cov_sims(gene_list=[50], pop_list=[2, 4, 8, 16, 32, 64], out_name = 'ba_cov_N_sims')
#run_ba_cov_sims(gene_list=[8, 16, 32, 64, 128], pop_list=[50], out_name = 'ba_cov_G_sims')
run_ba_cov_neutral_sims()
#rndm_sample_tenaillon()

# write code to re-shuffle proportion of positive/negative correlations

#run_ba_cor_sub_sims()
