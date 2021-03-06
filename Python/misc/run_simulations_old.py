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

import multiprocessing as mp



def time_partition_ltee(k=5, iter=1000):
    df_path = mydir + '/data/Good_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    to_include = pt.complete_nonmutator_lines()
    df_nonmut = df[df.index.str.contains('|'.join( to_include))]
    # remove columns with all zeros
    df_nonmut = df_nonmut.loc[:, (df_nonmut != 0).any(axis=0)]
    # make sure it's sorted
    df_nonmut.sort_index(inplace=True)

    time_points = [ int(x.split('_')[1]) for x in df_nonmut.index.values]
    time_points_set = sorted(list(set([ int(x.split('_')[1]) for x in df_nonmut.index.values])))
    time_points_positions = {}
    for x in time_points_set:
        time_points_positions[x] =  [i for i,j in enumerate(time_points) if j == x]
    t_final_df = df_nonmut.iloc[time_points_positions[max(time_points_set)]]
    t_final_np = t_final_df.values
    gene_names = df_nonmut.columns.tolist()

    df_out = open(mydir + '/data/Good_et_al/time_partition_z_scores.txt', 'w')
    df_out.write('\t'.join(['Time', 'less_mbd', 'greater_mpd', 'delta_mpd', 'less_mbd_025', 'less_mbd_975', 'greater_mpd_025', 'greater_mpd_975', 'delta_mpd_025', 'delta_mpd_975']) + '\n')
    for time_point in time_points_set:
        # very few mutations after generation 50000
        if time_point > 50000:
            continue
        print("Time point " + str(time_point))
        t_i_df = df_nonmut.iloc[time_points_positions[time_point]]
        t_i_np = t_i_df.values
        # remove rows with all zeros
        t_i_np_zeros = np.where(~t_i_np.any(axis=1))[0]
        n_zeros_t_i_np = len(t_i_np_zeros)
        if n_zeros_t_i_np > 0:
            t_i_np = np.delete(t_i_np, t_i_np_zeros, axis=0)

        t_i_to_final_np = t_final_np  - t_i_np
        # remove rows with all zeros
        t_i_to_final_np_zeros = np.where(~t_i_to_final_np.any(axis=1))[0]
        n_zeros_t_i_to_final_np = len(t_i_to_final_np_zeros)
        if n_zeros_t_i_to_final_np > 0:
            t_i_to_final_np = np.delete(t_i_to_final_np, t_i_to_final_np_zeros, axis=0)

        t_concat = np.concatenate((t_i_np, t_i_to_final_np), axis=0)
        t_norm = cd.likelihood_matrix_array(t_concat, gene_names, 'Good_et_al').get_likelihood_matrix()
        t_norm_rel = t_norm/t_norm.sum(axis=1, keepdims=True)
        t_norm_rel -= np.mean(t_norm_rel, axis = 0)
        pca = PCA()
        t_norm_rel_pca = pca.fit_transform(t_norm_rel)
        t_norm_rel_pca_k5 = t_norm_rel_pca[:, -1-k:-1]
        # account for rows with zero mutations
        dist_t_less = pt.get_mean_pairwise_euc_distance(t_norm_rel_pca_k5[:5-n_zeros_t_i_np,:], k=k)
        dist_t_greater = pt.get_mean_pairwise_euc_distance(t_norm_rel_pca_k5[5-n_zeros_t_i_to_final_np:,:], k=k)
        dist_t_change = dist_t_greater - dist_t_less
        #F_t = pt.get_F_2(t_norm_rel_pca_k5, 5-n_zeros_t_i_np, 5-n_zeros_t_i_to_final_np)[0]
        dist_t_less_list = []
        dist_t_greater_list = []
        dist_t_change_list = []
        #F_t_list = []
        for i in range(iter):
            if i % 1000 ==0:
                print("Iteration " + str(i))
            t_i_np_rndm = pt.get_random_matrix(t_i_np)
            t_i_to_final_np_rndm = pt.get_random_matrix(t_i_to_final_np)
            t_rndm_concat = np.concatenate((t_i_np_rndm, t_i_to_final_np_rndm), axis=0)
            t_rndm_norm = cd.likelihood_matrix_array(t_rndm_concat, gene_names, 'Good_et_al').get_likelihood_matrix()
            t_rndm_norm_rel = t_rndm_norm/t_rndm_norm.sum(axis=1, keepdims=True)
            t_rndm_norm_rel -= np.mean(t_rndm_norm_rel, axis = 0)
            t_rndm_norm_rel_pca = pca.fit_transform(t_rndm_norm_rel)
            # first five axes
            t_rndm_norm_rel_pca_k5 = t_rndm_norm_rel_pca[:, -1-k:-1]
            dist_t_less_rndm = pt.get_mean_pairwise_euc_distance(t_rndm_norm_rel_pca_k5[:5-n_zeros_t_i_np,:], k=k)
            dist_t_greater_rndm = pt.get_mean_pairwise_euc_distance(t_rndm_norm_rel_pca_k5[5-n_zeros_t_i_to_final_np:,:], k=k)
            dist_t_change_list.append(dist_t_greater_rndm - dist_t_less_rndm)
            dist_t_less_list.append(dist_t_less_rndm)
            dist_t_greater_list.append(dist_t_greater_rndm)
            #F_t_list.append(pt.get_F_2(t_rndm_norm_rel_pca, 5-n_zeros_t_i_np, 5-n_zeros_t_i_to_final_np)[0])

        dist_t_change_list.sort()
        dist_t_greater_list.sort()
        dist_t_less_list.sort()
        #F_t_list.sort()
        # get 95% CIs
        dist_t_change_025 = dist_t_change_list[int(iter*0.025)]
        dist_t_change_975 = dist_t_change_list[int(iter*0.975)]
        dist_t_greater_025 = dist_t_greater_list[int(iter*0.025)]
        dist_t_greater_975 = dist_t_greater_list[int(iter*0.975)]
        dist_t_less_025 = dist_t_less_list[int(iter*0.025)]
        dist_t_less_975 = dist_t_less_list[int(iter*0.975)]
        #F_t_025 = F_t_list[int(iter*0.025)]
        #F_t_975 = F_t_list[int(iter*0.975)]
        df_out.write('\t'.join([str(time_point), str(dist_t_less), str(dist_t_greater), \
                                str(dist_t_change), str(dist_t_less_025), str(dist_t_less_975), \
                                str(dist_t_greater_025), str(dist_t_greater_975), \
                                str(dist_t_change_025), str(dist_t_change_975)]) + '\n')

    df_out.close()





# Figure 1 code
# z = 1.645 for one sided test with alpha=0.05
def run_cov_sims(gene_list, pop_list, out_name, covs = [0.1, 0.15, 0.2], iter1=1000, iter2=1000):
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



def run_cov_neutral_sims(out_name, covs = [0.1, 0.15, 0.2], shape=1, scale=1, G = 50, N = 50, iter1=1000, iter2=1000):
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



def run_cov_prop_sims(out_name, covs = [0.1, 0.15, 0.2], props=[0.5], shape=1, scale=1, G = 50, N = 50, iter1=1000, iter2=1000):
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



def run_cov_rho_sims(out_name, covs = [0.1, 0.15, 0.2], rhos=[-0.2, 0, 0.2], shape=1, scale=1, G = 50, N = 50, iter1=10, iter2=1000):
    df_out=open(out_name, 'w')
    df_out.write('\t'.join(['N', 'G', 'Cov', 'Rho_goal', 'Rho_estimated', 'Iteration', 'dist_percent', 'z_score']) + '\n')
    for cov in covs:
        for rho in rhos:
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
                print(N, G, cov, rho, rho_estimated, i)
    df_out.close()





'''
Two treatments sims
'''

def run_cov_dist_sims(
    out_name,
    to_reshuffle = [5],
    N1=20,
    N2=20,
    covs=[0.05],
    G=100,
    shape = 1,
    scale = 1,
    iter1=10,
    iter2=1000):
    df_out=open(out_name, 'w')
    df_out.write('\t'.join(['N1', 'N2', 'G', 'Reshuf', 'Cov', 'Iteration', 'Euc_dist', 'F_2_percent', 'F_2_z_score', 'V_1_percent', 'V_1_z_score', 'V_2_percent', 'V_2_z_score']) + '\n')
    for reshuf in to_reshuffle:
        for cov in covs:
            reshuf_list = []
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
                euc_dist = np.linalg.norm(rates1-rates2)
                count_matrix = np.concatenate((counts1, counts2), axis=0)
                # check and remove empty columns
                count_matrix = count_matrix[:, ~np.all(count_matrix == 0, axis=0)]
                F_2_percent, F_2_z_score, \
                    V_1_percent, V_1_z_score, \
                    V_2_percent, V_2_z_score = \
                    pt.matrix_vs_null_two_treats(count_matrix,  N1, N2, iter=iter2)
                reshuf_list.append(euc_dist)
                print(reshuf, cov, i, F_2_percent, F_2_z_score, euc_dist, V_1_percent, V_2_percent)
                df_out.write('\t'.join([str(N1), str(N2), str(G), str(reshuf), str(cov), str(i), str(euc_dist), str(F_2_percent), str(F_2_z_score), str(V_1_percent), str(V_1_z_score), str(V_2_percent), str(V_2_z_score)]) + '\n')
            print(cov, np.mean(reshuf_list))
    df_out.close()



def run_cov_dist_sims_unequal(
    out_name,
    to_reshuffle = [5],
    N1=20,
    N2=20,
    covs_12=[0.05],
    G=100,
    shape = 1,
    scale = 1,
    iter1=10,
    iter2=1000):
    df_out=open(out_name, 'w')
    df_out.write('\t'.join(['N1', 'N2', 'G', 'Reshuf', 'Cov', 'Iteration', 'Euc_dist', 'F_2_percent', 'F_2_z_score', 'V_1_percent', 'V_1_z_score', 'V_2_percent', 'V_2_z_score']) + '\n')

    # re write code for covariance matrix to get
    for reshuf in to_reshuffle:
        for cov in covs:
            reshuf_list = []
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
                euc_dist = np.linalg.norm(rates1-rates2)
                count_matrix = np.concatenate((counts1, counts2), axis=0)
                # check and remove empty columns
                count_matrix = count_matrix[:, ~np.all(count_matrix == 0, axis=0)]
                F_2_percent, F_2_z_score, \
                    V_1_percent, V_1_z_score, \
                    V_2_percent, V_2_z_score = \
                    pt.matrix_vs_null_two_treats(count_matrix,  N1, N2, iter=iter2)
                reshuf_list.append(euc_dist)
                print(reshuf, cov, i, F_2_percent, F_2_z_score, euc_dist, V_1_percent, V_2_percent)
                df_out.write('\t'.join([str(N1), str(N2), str(G), str(reshuf), str(cov), str(i), str(euc_dist), str(F_2_percent), str(F_2_z_score), str(V_1_percent), str(V_1_z_score), str(V_2_percent), str(V_2_z_score)]) + '\n')
            print(cov, np.mean(reshuf_list))
    df_out.close()



def time_partition_ltee(k=5, iter=100):
    df_path = os.path.expanduser("~/GitHub/ParEvol") + '/data/Good_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    to_include = pt.complete_nonmutator_lines()
    df_nonmut = df[df.index.str.contains('|'.join( to_include))]
    # remove columns with all zeros
    df_nonmut = df_nonmut.loc[:, (df_nonmut != 0).any(axis=0)]
    # make sure it's sorted
    df_nonmut.sort_index(inplace=True)

    time_points = [ int(x.split('_')[1]) for x in df_nonmut.index.values]
    time_points_set = sorted(list(set([ int(x.split('_')[1]) for x in df_nonmut.index.values])))
    time_points_positions = {}
    for x in time_points_set:
        time_points_positions[x] =  [i for i,j in enumerate(time_points) if j == x]
    t_final_df = df_nonmut.iloc[time_points_positions[max(time_points_set)]]
    t_final_np = t_final_df.values
    gene_names = df_nonmut.columns.tolist()

    df_out = open(os.path.expanduser("~/GitHub/ParEvol") + '/data/Good_et_al/time_partition_z_scores.txt', 'w')
    df_out.write('\t'.join(['Time', 'Time_less_z_score', 'Time_greater_z_score']) + '\n')
    for time_point in time_points_set:
        # very few mutations after generation 50000
        if time_point > 50000:
            continue
        print("Time point " + str(time_point))
        t_i_df = df_nonmut.iloc[time_points_positions[time_point]]
        t_i_np = t_i_df.values
        # remove rows with all zeros
        t_i_np_zeros = np.where(~t_i_np.any(axis=1))[0]
        n_zeros_t_i_np = len(t_i_np_zeros)
        if n_zeros_t_i_np > 0:
            t_i_np = np.delete(t_i_np, t_i_np_zeros, axis=0)

        t_i_to_final_np = t_final_np  - t_i_np
        # remove rows with all zeros
        t_i_to_final_np_zeros = np.where(~t_i_to_final_np.any(axis=1))[0]
        n_zeros_t_i_to_final_np = len(t_i_to_final_np_zeros)
        if n_zeros_t_i_to_final_np > 0:
            t_i_to_final_np = np.delete(t_i_to_final_np, t_i_to_final_np_zeros, axis=0)

        t_concat = np.concatenate((t_i_np, t_i_to_final_np), axis=0)
        t_norm = cd.likelihood_matrix_array(t_concat, gene_names, 'Good_et_al').get_likelihood_matrix()
        t_norm_rel = t_norm/t_norm.sum(axis=1, keepdims=True)

        # first five axes
        e_vals, e_vecs = pt.pca_np(t_norm_rel)
        # The column v[:, i] is the normalized eigenvector corresponding to the eigenvalue w[i]
        e_vecs_k5 = e_vecs[:, -1-k:-1]
        # account for rows with zero mutations
        e_vec_t_less = e_vecs_k5[:5-n_zeros_t_i_np,:]
        e_vec_t_greater = e_vecs_k5[5-n_zeros_t_i_to_final_np:,:]

        dist_t_less = pt.get_mean_pairwise_euc_distance(e_vec_t_less, k=k)
        dist_t_greater = pt.get_mean_pairwise_euc_distance(e_vec_t_greater, k=k)

        dist_t_less_list = []
        dist_t_greater_list = []

        for i in range(iter):
            t_i_np_rndm = pt.get_random_matrix(t_i_np)
            t_i_to_final_np_rndm = pt.get_random_matrix(t_i_to_final_np)
            t_rndm_concat = np.concatenate((t_i_np_rndm, t_i_to_final_np_rndm), axis=0)

            t_rndm_norm = cd.likelihood_matrix_array(t_rndm_concat, gene_names, 'Good_et_al').get_likelihood_matrix()
            t_rndm_norm_rel = t_rndm_norm/t_rndm_norm.sum(axis=1, keepdims=True)
            # first five axes
            e_vals_rndm, e_vecs_rndm = pt.pca_np(t_rndm_norm_rel)
            # The column v[:, i] is the normalized eigenvector corresponding to the eigenvalue w[i]
            e_vecs_rndm_k5 = e_vecs_rndm[:, -1-k:-1]
            e_vec_t_less_rndm = e_vecs_rndm_k5[:5,:]
            e_vec_t_greater_rndm = e_vecs_rndm_k5[5:,:]

            dist_t_less_rndm = pt.get_mean_pairwise_euc_distance(e_vec_t_less_rndm, k=k)
            dist_t_greater_rndm = pt.get_mean_pairwise_euc_distance(e_vec_t_greater_rndm, k=k)

            dist_t_less_list.append(dist_t_less_rndm)
            dist_t_greater_list.append(dist_t_greater_rndm)

        z_score_less = (dist_t_less - np.mean(dist_t_less_list)) / np.std(dist_t_less_list)
        z_score_greater = (dist_t_greater - np.mean(dist_t_greater_list)) / np.std(dist_t_greater_list)

        df_out.write('\t'.join([str(time_point), str(z_score_less), str(z_score_greater)]) + '\n')

    df_out.close()










def permute_ltee(k = 5, n_blocks = 2):
    df_path = os.path.expanduser("~/GitHub/ParEvol") + '/data/Good_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    to_include = pt.complete_nonmutator_lines()
    to_keep.remove('p5')
    df_nonmut = df[df.index.str.contains('|'.join( to_exclude))]
    # remove columns with all zeros
    df_nonmut = df_nonmut.loc[:, (df_nonmut != 0).any(axis=0)]
    time_points = [ int(x.split('_')[1]) for x in df_nonmut.index.values]
    time_points_set = sorted(list(set([ int(x.split('_')[1]) for x in df_nonmut.index.values])))
    df_nonmut_array = df_nonmut.as_matrix()
    time_points_positions = {}
    for x in time_points_set:
        time_points_positions[x] =  [i for i,j in enumerate(time_points) if j == x]
    df_final = df_nonmut.iloc[time_points_positions[time_points_set[-1]]]

    df_out = open(pt.get_path() + '/data/Good_et_al/permute_' + analysis + '.txt', 'w')
    #column_headers = ['Iteration', 'Generation', 'MCD']
    column_headers = ['Iteration', 'Generation', 'mean_dist']
    df_out.write('\t'.join(column_headers) + '\n')
    for i in range(iter):
        continue
        print("Iteration " + str(i))
        matrix_0 = df_nonmut.iloc[time_points_positions[time_points_set[0]]]
        matrix_0_rndm = pt.random_matrix(matrix_0.as_matrix())
        df_rndm_list = [pd.DataFrame(data=matrix_0_rndm, index=matrix_0.index, columns=matrix_0.columns)]
        # skip first time step
        for j, tp in enumerate(time_points_set[0:]):
            if j == 0:
                continue
            df_tp_minus1 = df_nonmut[df_nonmut.index.str.contains('_' + str(time_points_set[j-1]))]
            df_tp = df_nonmut[df_nonmut.index.str.contains('_' + str(tp))]
            matrix_diff = df_tp.as_matrix() - df_tp_minus1.as_matrix()
            matrix_0_rndm = matrix_0_rndm +  pt.random_matrix(matrix_diff)
            df_0_rndm = pd.DataFrame(data=matrix_0_rndm, index=df_tp.index, columns=df_tp.columns)
            df_rndm_list.append(df_0_rndm)

        df_rndm = pd.concat(df_rndm_list)
        df_rndm_delta = pt.likelihood_matrix(df_rndm, 'Good_et_al').get_likelihood_matrix()
        if analysis == 'PCA':
            X = pt.hellinger_transform(df_rndm_delta)
            pca = PCA()
            matrix_rndm_delta_out = pca.fit_transform(X)
        elif analysis == 'cMDS':
            matrix_rndm_delta_bc = np.sqrt(pt.get_bray_curtis(df_rndm_delta.as_matrix()))
            matrix_rndm_delta_out = pt.cmdscale(matrix_rndm_delta_bc)[0]
        else:
            print("Analysis argument not accepted")
            continue

        df_rndm_delta_out = pd.DataFrame(data=matrix_rndm_delta_out, index=df_rndm_delta.index)
        for tp in time_points_set:
            df_rndm_delta_out_tp = df_rndm_delta_out[df_rndm_delta_out.index.str.contains('_' + str(tp))]
            df_rndm_delta_out_tp_matrix = df_rndm_delta_out_tp.as_matrix()
            mean_angle = pt.get_mean_angle(df_rndm_delta_out_tp_matrix, k = k)
            mcd = pt.get_mean_centroid_distance(df_rndm_delta_out_tp_matrix, k=k)
            mean_length = pt.get_euc_magnitude_diff(df_rndm_delta_out_tp_matrix, k=k)
            mean_dist = pt.get_mean_pairwise_euc_distance(df_rndm_delta_out_tp_matrix, k=k)
            df_out.write('\t'.join([str(i), str(tp), str(mcd), str(mean_angle), str(mean_length), str(mean_dist) ]) + '\n')

    df_out.close()






def rndm_sample_tenaillon(iter1=1000, iter2=10000):
    df_path = os.path.expanduser("~/GitHub/ParEvol") + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    n_rows = list(range(df_np.shape[0]))
    df_out=open(os.path.expanduser("~/GitHub/ParEvol") + '/data/Tenaillon_et_al/dist_sample_size.txt', 'w')
    df_out.write('\t'.join(['N', 'G', 'Iteration', 'dist_percent', 'z_score']) + '\n')
    #Ns = list(range(4, 40 +2, 2))
    Ns = [40]
    pca = PCA()
    for N in Ns:
        for i in range(iter1):
            df_np_i = df_np[np.random.choice(n_rows, size=N, replace=False, p=None), :]
            gene_bool = np.all(df_np_i == 0, axis=0)
            # flip around to select gene_size
            gene_names_i = list(compress(gene_names, list(map(operator.not_, gene_bool))))
            df_np_i = df_np_i[:,~np.all(df_np_i == 0, axis=0)]
            np.seterr(divide='ignore')
            df_np_i_delta = cd.likelihood_matrix_array(df_np_i, gene_names_i, 'Tenaillon_et_al').get_likelihood_matrix()
            df_np_i_delta = df_np_i_delta/df_np_i_delta.sum(axis=1)[:,None]
            X = pt.get_mean_center(df_np_i_delta)
            pca_fit = pca.fit_transform(X)
            euc_dist = pt.get_mean_pairwise_euc_distance(pca_fit)
            euc_dists = []
            for j in range(iter2):
                df_np_i_j = pt.get_random_matrix(df_np_i)
                np.seterr(divide='ignore')
                df_np_i_j_delta = cd.likelihood_matrix_array(df_np_i_j, gene_names_i, 'Tenaillon_et_al').get_likelihood_matrix()
                df_np_i_j_delta = df_np_i_j_delta/df_np_i_j_delta.sum(axis=1)[:,None]
                X_j = pt.get_mean_center(df_np_i_j_delta)
                pca_fit_j = pca.fit_transform(X_j)
                euc_dists.append( pt.get_mean_pairwise_euc_distance(pca_fit_j) )

            G = df_np_i.shape[1]
            euc_percent = len( [k for k in euc_dists if k > euc_dist] ) / len(euc_dists)
            z_score = (euc_dist - np.mean(euc_dists)) / np.std(euc_dists)
            print(str(N), str(i), str(G), str(euc_percent), str(z_score))
            df_out.write('\t'.join([str(N), str(G), str(i), str(euc_percent), str(z_score)]) + '\n')

    df_out.close()









def gene_svd_tenaillon(iter=10000):
    df_path = os.path.expanduser("~/GitHub/ParEvol") + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    gene_names = df.columns.tolist()
    df_np = df.values
    df_np_delta = cd.likelihood_matrix_array(df_np, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()
    df_np_delta = df_np_delta/df_np_delta.sum(axis=1)[:,None]
    X = pt.get_mean_center(df_np_delta)
    # scipy's svd returns the V matrix in transposed form
    U, s, V_T = svds(X, k=3)
    # apply another transposition to calculate basis matrix
    F = (V_T.T @ np.diag(s)) / np.sqrt(  X.shape[0] - 1 )
    vars = np.linalg.norm(F, axis=1) ** 2
    vars_null_list = []
    for i in range(iter):
        if i % 1000 ==0:
            print("Iteration " + str(i))
        df_np_i = pt.get_random_matrix(df_np)
        np.seterr(divide='ignore')
        df_np_i_delta = cd.likelihood_matrix_array(df_np_i, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()
        df_np_i_delta = df_np_i_delta/df_np_i_delta.sum(axis=1)[:,None]
        X_j = pt.get_mean_center(df_np_i_delta)
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
#rndm_sample_tenaillon()
#gene_svd_tenaillon()
#gene_svd_tenaillon_sample_size()

#permute_ltee()
time_partition_ltee()


















def rndm_sample_tenaillon(N, df_np, gene_names, n_rows, k=3, iter1=100, iter2=1000, sample_bs = 10, iter_bs=10000):
    p_values = []
    #z_scores = []
    G_list = []
    for i in range(iter1):
        df_np_i = df_np[np.random.choice(n_rows, size=N, replace=False, p=None), :]
        gene_bool = np.all(df_np_i == 0, axis=0)
        # flip around to select gene_size
        gene_names_i = list(compress(gene_names, list(map(operator.not_, gene_bool))))
        G_list.append(len(gene_names_i))
        df_np_i = df_np_i[:,~np.all(df_np_i == 0, axis=0)]
        np.seterr(divide='ignore')
        df_np_i_delta = cd.likelihood_matrix_array(df_np_i, gene_names_i, 'Tenaillon_et_al').get_likelihood_matrix()
        X = df_np_i_delta/df_np_i_delta.sum(axis=1)[:,None]
        e_vals, a_mat = pt.pca_np(X)
        euc_dist = pt.get_mean_pairwise_euc_distance(a_mat, k=k)
        print(euc_dist)
        euc_dists = []
        for j in range(iter2):
            df_np_i_j = pt.get_random_matrix(df_np_i)
            np.seterr(divide='ignore')
            df_np_i_j_delta = cd.likelihood_matrix_array(df_np_i_j, gene_names_i, 'Tenaillon_et_al').get_likelihood_matrix()
            X_j = df_np_i_j_delta/df_np_i_j_delta.sum(axis=1)[:,None]
            e_vals_j, a_mat_j = pt.pca_np(X_j)
            euc_dists.append(pt.get_mean_pairwise_euc_distance(a_mat_j, k=k))
        p_values.append(len( [m for m in euc_dists if m > euc_dist] ) / len(euc_dists))
        #z_scores.append( (euc_dist - np.mean(euc_dists)) / np.std(euc_dists) )s

    power = len([n for n in p_values if n < 0.05]) / len(p_values)
    print(p_values)
    power_bootstrap = []
    for p in range(iter_bs):
        p_values_sample = random.sample(p_values, sample_bs)
        power_sample = len([n for n in p_values_sample if n < 0.05]) / len(p_values_sample)
        power_bootstrap.append(power_sample)
    power_bootstrap.sort()
    # return number of genes, power, power lower, power upper
    return N, np.mean(G_list), power, power_bootstrap[int(10000*0.025)], power_bootstrap[int(10000*0.975)]





def parallel_rndm_sample_tenaillon(k=3, iter1=20, iter2=1000):
    df_path = os.path.expanduser("~/GitHub/ParEvol") + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    n_rows = list(range(df_np.shape[0]))

    Ns = [20, 30]
    #Ns = list(range(20, n_rows, 4))


    pool = mp.Pool(processes=4)


    rndm_sample_tenaillon_partial = partial(rndm_sample_tenaillon, df_np=df_np, gene_names=gene_names, n_rows=n_rows, k=k, iter1=iter1, iter2=iter2) # prod_x has only one argument x (y is fixed to 10)
    result_list = pool.map(rndm_sample_tenaillon_partial, Ns)
    print(result_list)

    df_out = open(os.path.expanduser("~/GitHub/ParEvol") + '/data/Tenaillon_et_al/dist_sample_size.txt', 'w')
    df_out.write('\t'.join(['N', 'G', 'Power', 'Power_025', 'Power_975']) + '\n')


parallel_rndm_sample_tenaillon()
