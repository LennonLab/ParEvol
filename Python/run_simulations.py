from __future__ import division
import os, pickle, operator
from random import shuffle
from itertools import compress
from collections import Counter
import numpy as np
import pandas as pd
import parevol_tools as pt
import clean_data as cd
from sklearn.decomposition import PCA
import scipy.stats as ss
from scipy.sparse.linalg import svds

# Figure 1 code
# z = 1.645 for one sided test with alpha=0.05
def run_ba_cov_sims(gene_list, pop_list, out_name, covs = [0.1, 0.15, 0.2], iter1=1000, iter2=1000):
    df_out=open(out_name, 'w')
    df_out.write('\t'.join(['N', 'G', 'Cov', 'Iteration', 'dist_percent', 'z_score']) + '\n')
    for G in gene_list:
        for N in pop_list:
            for cov in covs:
                for i in range(iter1):
                    C = pt.get_ba_cov_matrix(G, cov)
                    while True:
                        lambda_genes = np.random.gamma(shape=1, scale=1, size=G)
                        test_cov = np.stack( [pt.get_count_pop(lambda_genes, C= C) for x in range(N)] , axis=0 )
                        if (np.any(test_cov.sum(axis=1) == 0 )) == False:
                            break
                    # check and remove empty columns
                    test_cov = test_cov[:, ~np.all(test_cov == 0, axis=0)]
                    euc_percent, z_score = pt.matrix_vs_null_one_treat(test_cov, iter2)
                    df_out.write('\t'.join([str(N), str(G), str(cov), str(i), str(euc_percent), str(z_score)]) + '\n')
                print(N, G, cov)
    df_out.close()



def run_ba_cov_neutral_sims(out_name, covs = [0.1, 0.15, 0.2], shape=1, scale=1, G = 50, N = 50, iter1=1000, iter2=1000):
    df_out=open(out_name, 'w')
    df_out.write('\t'.join(['N', 'G', 'lamba_mean', 'lambda_neutral', 'Cov', 'Iteration', 'dist_percent', 'z_score']) + '\n')
    mean_gamma = shape * scale
    neutral_range = np.logspace(-2, 1, num=20, endpoint=True, base=10.0)
    neutral_range = neutral_range[::-1]
    for neutral_ in neutral_range:
        for cov in covs:
            for i in range(iter1):
                C = pt.get_ba_cov_matrix(G, cov)
                while True:
                    lambda_genes = np.random.gamma(shape=shape, scale=scale, size=G)
                    lambda_genes_null = np.asarray([neutral_] * G)
                    test_cov_adapt = np.stack( [pt.get_count_pop(lambda_genes, C= C) for x in range(N)] , axis=0 )
                    # matrix with diaganol values equal to one
                    test_cov_neutral = np.stack( [pt.get_count_pop(lambda_genes_null, C= np.identity(G)) for x in range(N)] , axis=0 )
                    test_cov = test_cov_adapt + test_cov_neutral
                    if (np.any(test_cov.sum(axis=1) == 0 )) == False:
                        break
                # check and remove empty columns
                test_cov = test_cov[:, ~np.all(test_cov == 0, axis=0)]
                euc_percent, z_score = pt.matrix_vs_null_one_treat(test_cov, iter2)
                df_out.write('\t'.join([str(N), str(G), str(mean_gamma), str(neutral_), str(cov), str(i), str(euc_percent), str(z_score)]) + '\n')
            print(neutral_, cov)
    df_out.close()



def run_ba_cov_prop_sims(out_name, covs = [0.1, 0.15, 0.2], props=[0.5], shape=1, scale=1, G = 50, N = 50, iter1=1000, iter2=1000):
    df_out=open(out_name, 'w')
    df_out.write('\t'.join(['N', 'G', 'Cov', 'Proportion', 'Iteration', 'dist_percent', 'z_score']) + '\n')
    for prop in props:
        for cov in covs:
            for i in range(iter1):
                C = pt.get_ba_cov_matrix(G, cov, prop=prop)
                while True:
                    lambda_genes = np.random.gamma(shape=1, scale=1, size=G)
                    test_cov = np.stack( [pt.get_count_pop(lambda_genes, C= C) for x in range(N)] , axis=0 )
                    if (np.any(test_cov.sum(axis=1) == 0 )) == False:
                        break
                # check and remove empty columns
                test_cov = test_cov[:, ~np.all(test_cov == 0, axis=0)]
                euc_percent, z_score = pt.matrix_vs_null_one_treat(test_cov, iter2)
                df_out.write('\t'.join([str(N), str(G), str(cov), str(prop), str(i), str(euc_percent), str(z_score)]) + '\n')
            print(N, G, cov)
    df_out.close()



def run_ba_cov_rho_sims(out_name, covs = [0.1, 0.15, 0.2], rhos=[0.2], shape=1, scale=1, G = 50, N = 50, iter1=1000, iter2=1000):
    df_out=open(out_name, 'w')
    df_out.write('\t'.join(['N', 'G', 'Cov', 'Rho_goal', 'Rho_estimated', 'Iteration', 'dist_percent', 'z_score']) + '\n')
    for rho in rhos:
        for cov in covs:
            for i in range(iter1):
                C, rho_estimated = pt.get_ba_cov_matrix(n_genes=G, cov=cov, rho=rho)
                while True:
                    lambda_genes = np.random.gamma(shape=1, scale=1, size=G)
                    test_cov = np.stack( [pt.get_count_pop(lambda_genes, C= C) for x in range(N)] , axis=0 )
                    if (np.any(test_cov.sum(axis=1) == 0 )) == False:
                        break
                # check and remove empty columns
                test_cov = test_cov[:, ~np.all(test_cov == 0, axis=0)]
                euc_percent, z_score = pt.matrix_vs_null_one_treat(test_cov, iter2)
                df_out.write('\t'.join([str(N), str(G), str(cov), str(rho), str(rho_estimated), str(i), str(euc_percent), str(z_score)]) + '\n')
            print(N, G, cov)
    df_out.close()



def run_ba_cov_lampbda_edge_sims(out_name, num_permute, G=50,shape=1, scale=1):
    rates = np.random.gamma(shape, scale=scale, size=G)
    cov=0.2
    C = pt.get_ba_cov_matrix(10, cov)
    print(C)
    counts = np.count_nonzero(C,axis=1).flatten()
    counts = np.asarray(counts.tolist()[0])
    print(len(counts))
    #order_C = np.asarray(np.count_nonzero(C,axis=1).tolist()[0])
    #print(order_C)

    #print(np.linalg.norm(C,axis=1))
    #print(np.argsort(np.linalg.norm(C,axis=1)))

    #corrs = C.sum(axis=0)
    #corr_order = corrs.argsort()[::-1]
    #print(np.count_nonzero(C,axis=1))

    #ndim = C.shape[0]
    #inds_orig = list(range(ndim))
    #inds = []
    #for _ in range(ndim):
    #    inds.append(inds_orig[(len(inds_orig)-1)//2])
    #    del inds_orig[(len(inds_orig)-1)//2]
    #inds = np.array(inds)
    #res = np.empty_like(C)
    #corr_order_flatten = np.asarray(corr_order.tolist()[0])
    #res[np.ix_(inds,inds)] = C[np.ix_(corr_order_flatten,corr_order_flatten)]

    #print(np.count_nonzero(res,axis=1))

    #print(np.count_nonzero(C, axis=1))
    #ranked = ss.rankdata(np.count_nonzero(C, axis=1))
    #print(ranked)
    #diag_C = np.tril(C, k =-1)
    #i,j = np.nonzero(diag_C)
    # remove redundant pairs
    #ix = np.random.choice(len(i), int(np.floor((1-prop) * len(i))), replace=False)







'''
two treatments sims
'''


def two_treats_sim(
    to_reshuffle = [10],
    N1=10,
    N2=10,
    covs=[0.05],
    G=100,
    shape = 1,
    scale = 1,
    iter1=100,
    iter2=1000):

    for reshuf in to_reshuffle:
        for cov in covs:
            F_2_list = [ ]
            for i in range(iter1):
                C = pt.get_ba_cov_matrix(G, cov)
                while True:
                    rates = np.random.gamma(shape, scale=scale, size=G)
                    rates1 = rates.copy()
                    rates2 = rates.copy()
                    # fix this so you're not resampling the same pairs
                    for j in range(reshuf)[0::2]:
                        rates2[j], rates2[j+1] = rates2[j+1], rates2[j]
                    #shuffle(rates)#[:reshuf])
                    counts1 = np.stack( [pt.get_count_pop(rates1, C= C) for x in range(N1)], axis=0)
                    counts2 = np.stack( [pt.get_count_pop(rates2, C= C) for x in range(N2)], axis=0)
                    if (np.any(counts1.sum(axis=1) == 0 ) == False) or (np.any(counts2.sum(axis=1) == 0 ) == False):
                        break
                #D_KL = ss.entropy(rates1, rates2)
                euc_dist = np.linalg.norm(rates1-rates2)
                count_matrix = np.concatenate((counts1, counts2), axis=0)
                # check and remove empty columns
                count_matrix = count_matrix[:, ~np.all(count_matrix == 0, axis=0)]
                F_2_percent, F_2_z_score, V_1_percent, \
                    V_1_z_score, V_2_percent, V_2_z_score = \
                    pt.matrix_vs_null_two_treats(count_matrix,  N1, N2, iter=iter2)

                print(i, F_2_percent, F_2_z_score, euc_dist)
                F_2_list.append(F_2_percent)

            p_sig = [p_i for p_i in F_2_list if p_i >= (1-0.05)]
            print(len(p_sig) / len(F_2_list))







#def run_ba_cor_sub_sims(shape=1, scale=1, G = 50, N = 50):
#    cov = 0.1
#    lambda_genes = np.random.gamma(shape=shape, scale=scale, size=G)
#    lambda_genes.sort()
#    C, edge_count = pt.get_ba_cov_matrix(G, cov, get_node_edge_sum=True)
#    zipped = list(zip( list(range(G)) , edge_count.tolist()[0] ))
#    zipped.sort(key = lambda t: t[1])
#    # figure out how to sort covariance matrix
#    total_inversions = 30
#    inversion_count = 0
#    while inversion_count < total_inversions:
#        pair = np.random.choice(list(range(G)), size = 2, replace=False)
#        pair.sort()
#        if lambda_genes[pair[0]] < lambda_genes[pair[1]]:
#            lambda_0 =lambda_genes[pair[0]].copy()
#            lambda_1 =lambda_genes[pair[1]].copy()
#            lambda_genes[pair[0]] = lambda_1
#            lambda_genes[pair[1]] = lambda_0
#            inversion_count += 1
#    unzipped = list(zip(*zipped))
#    rezipped = list( zip( unzipped[0], unzipped[1],  lambda_genes) )
#    rezipped.sort(key = lambda t: t[0])
#    unrezipped = list(zip(*rezipped))
#    lambda_genes_sorted = list(unrezipped[2])

#    print(lambda_genes_sorted)

    #list[min(lambda_genes[pair[0]],j)] < list[max(i,j)]
    #print()


def rndm_sample_tenaillon(iter1=1000, iter2=10000):
    df_path = os.path.expanduser("~/GitHub/ParEvol") + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    #n_rows = df_np.shape[0]
    n_rows = list(range(df_np.shape[0]))
    df_out=open(os.path.expanduser("~/GitHub/ParEvol") + '/data/Tenaillon_et_al/dist_sample_size.txt', 'w')
    df_out.write('\t'.join(['N', 'G', 'Iteration', 'dist_percent', 'z_score']) + '\n')
    Ns = list(range(4, 40, 2))
    for N in Ns:
        for i in range(iter1):
            #df_np_i = df_np[np.random.randint(n_rows, size=N), :]
            df_np_i = df_np[np.random.choice(n_rows, size=N, replace=False, p=None), :]
            gene_bool = np.all(df_np_i == 0, axis=0)
            # flip around to select gene_size
            gene_names_i = list(compress(gene_names, list(map(operator.not_, gene_bool))))
            df_np_i = df_np_i[:,~np.all(df_np_i == 0, axis=0)]
            np.seterr(divide='ignore')
            #df_np_i_delta = cd.likelihood_matrix_array(df_np_i, gene_names_i, 'Tenaillon_et_al').get_likelihood_matrix()
            X = pt.get_mean_center(df_np_i)
            pca = PCA()
            pca_fit = pca.fit_transform(X)
            euc_dist = pt.get_mean_pairwise_euc_distance(pca_fit)
            euc_dists = []
            for j in range(iter2):
                df_np_i_j = pt.get_random_matrix(df_np_i)
                np.seterr(divide='ignore')
                #df_np_i_j_delta = cd.likelihood_matrix_array(df_np_i_j, gene_names_i, 'Tenaillon_et_al').get_likelihood_matrix()
                X_j = pt.get_mean_center(df_np_i_j)
                pca_fit_j = pca.fit_transform(X_j)
                euc_dists.append( pt.get_mean_pairwise_euc_distance(pca_fit_j) )

            G = df_np_i.shape[1]
            euc_percent = len( [k for k in euc_dists if k < euc_dist] ) / len(euc_dists)
            z_score = (euc_dist - np.mean(euc_dists)) / np.std(euc_dists)
            print(str(N), str(i), str(G), str(euc_percent), str(z_score))
            df_out.write('\t'.join([str(N), str(G), str(i), str(euc_percent), str(z_score)]) + '\n')

    df_out.close()


def gene_svd_tenaillon(iter=10000):
    df_path = os.path.expanduser("~/GitHub/ParEvol") + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    X = pt.get_mean_center(df_np)
    # scipy's svd returns the V matrix in transposed form
    U, s, V_T = svds(X, k=3)
    # apply another transposition to calculate basis matrix
    F = (V_T.T @ np.diag(s)) / np.sqrt(  X.shape[0] - 1 )
    vars = np.linalg.norm(F, axis=1) ** 2
    gene_names = df.columns.tolist()
    vars_null_list = []
    for i in range(iter):
        if i % 1000 ==0:
            print("Iteration " + str(i))
        df_np_i = pt.get_random_matrix(df_np)
        np.seterr(divide='ignore')
        #df_np_i_j_delta = cd.likelihood_matrix_array(df_np_i_j, gene_names_i, 'Tenaillon_et_al').get_likelihood_matrix()
        X_j = pt.get_mean_center(df_np_i)
        U_i, s_i, V_i_T = svds(X_j, k=3)
        F_i = (V_i_T.T @ np.diag(s_i)) / np.sqrt(  X_j.shape[0] - 1 )
        vars_null_list.append(np.linalg.norm(F_i, axis=1) ** 2)

    vars_null = np.stack(vars_null_list)
    vars_null_mean = np.mean(vars_null, axis=0)
    vars_null_std = np.std(vars_null, axis=0)
    z_scores = (vars - vars_null_mean) / vars_null_std
    p_values = []
    # calculate p values
    for k, column in enumerate(vars_null.T):
        column_greater = [x for x in column if x > vars[k]]
        p_values.append(len(column_greater) / iter)
        #print(k, vars[k], len(column_greater) / iter)

    label_z_scores = list(zip(gene_names, z_scores, p_values))
    label_sig_z_scores = [x for x in label_z_scores if x[2] < 0.05]
    print(label_sig_z_scores)

    df_out=open(os.path.expanduser("~/GitHub/ParEvol") + '/data/Tenaillon_et_al/gene_z_scores.txt', 'w')
    df_out.write('\t'.join(['Gene', 'z_score', 'p_score']) + '\n')
    for label_z_score in label_z_scores:
        df_out.write('\t'.join([str(label_z_score[0]), str(label_z_score[1]), str(label_z_score[2])]) + '\n')
    df_out.close()




def gene_svd_tenaillon_sample_size(iter1 = 1000, iter2=10000, k =3):
    df_path =os.path.expanduser("~/GitHub/ParEvol") + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    n_rows = list(range(df_np.shape[0]))
    df_out=open(os.path.expanduser("~/GitHub/ParEvol") + '/data/Tenaillon_et_al/gene_z_scores_sample_size.txt', 'w')
    df_out.write('\t'.join(['N', 'G', 'Iteration', 'set_percent']) + '\n')
    Ns = list(range(4, 40, 2))
    # get genes with an absolute z-score greater than 1.96
    df_gene_path = os.path.expanduser("~/GitHub/ParEvol") + '/data/Tenaillon_et_al/gene_z_scores.txt'
    df_genes = pd.read_csv(df_gene_path, sep = '\t', header = 'infer')
    df_genes_sig = df_genes.loc[(df_genes['z_score'] > 1.96) | (df_genes['z_score'] < -1.96)]
    genes = df_genes_sig.Gene.tolist()
    for N in Ns:
        for i in range(iter1):
            df_np_i = df_np[np.random.choice(n_rows, size=N, replace=False, p=None), :]
            testtt = np.random.choice(n_rows, size=N, replace=False, p=None)
            gene_bool = np.all(df_np_i == 0, axis=0)
            # flip around to select gene_size
            gene_names_i = list(compress(gene_names, list(map(operator.not_, gene_bool))))
            df_np_i = df_np_i[:,~np.all(df_np_i == 0, axis=0)]
            np.seterr(divide='ignore')
            #df_np_i_delta = cd.likelihood_matrix_array(df_np_i, gene_names_i, 'Tenaillon_et_al').get_likelihood_matrix()
            X = pt.get_mean_center(df_np_i)
            U, s, V_T = svds(X, k=k)
            # apply another transposition to calculate basis matrix
            F = (V_T.T @ np.diag(s)) / np.sqrt(  X.shape[0] - 1 )
            vars = np.linalg.norm(F, axis=1) ** 2
            vars_null_list = []
            for j in range(iter2):
                df_np_i_j = pt.get_random_matrix(df_np_i)
                np.seterr(divide='ignore')
                #df_np_i_j_delta = cd.likelihood_matrix_array(df_np_i_j, gene_names_i, 'Tenaillon_et_al').get_likelihood_matrix()
                X_j = pt.get_mean_center(df_np_i_j)
                U_j, s_j, V_j_T = svds(X_j, k=3)
                F_j = (V_j_T.T @ np.diag(s_j)) / np.sqrt(  X_j.shape[0] - 1 )
                vars_null_list.append(np.linalg.norm(F_j, axis=1) ** 2)

            vars_null_i = np.stack(vars_null_list)
            vars_null_i_mean = np.mean(vars_null_i, axis=0)
            vars_null_i_std = np.std(vars_null_i, axis=0)
            z_scores = (vars - vars_null_i_mean) / vars_null_i_std
            label_z_scores = list(zip(gene_names_i, z_scores))
            label_sig_z_scores = [x for x in label_z_scores if abs(x[1]) > 1.96]
            label_sig_z_scores_label = [x[0] for x in label_sig_z_scores]
            gene_inter = set(label_sig_z_scores_label) & set(genes)
            union_fract = len(gene_inter) / len(genes)
            print(N, i, union_fract)
            G = df_np_i.shape[1]
            df_out.write('\t'.join([str(N), str(G), str(i), str(union_fract)]) + '\n')

    df_out.close()


#rndm_sample_tenaillon()
#gene_svd_tenaillon()
#gene_svd_tenaillon_sample_size()

# write code to re-shuffle proportion of positive/negative correlations

#run_ba_cor_sub_sims()
