from __future__ import division
import os, pickle, operator
import random
from itertools import compress
import numpy as np
import pandas as pd
import parevol_tools as pt
import clean_data as cd

#import multiprocessing as mp
#from functools import partial
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA


#mydir = '/N/dc2/projects/Lennon_Sequences/ParEvol'
mydir = os.path.expanduser("~/GitHub/ParEvol")


def rndm_sample_tenaillon(k_eval=3, iter1=20, iter2=1000, sample_bs = 10, iter_bs=10000):
    df_path = mydir + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    n_rows = list(range(df_np.shape[0]))
    df_out = open(mydir + '/data/Tenaillon_et_al/power_sample_size.txt', 'w')
    df_out.write('\t'.join(['N', 'G', 'Power', 'Power_025', 'Power_975']) + '\n')

    Ns = [20, 30]
    #Ns = list(range(20, n_rows, 4))
    for N in Ns:
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
            X -= np.mean(X, axis = 0)
            pca = PCA()
            pca_X = pca.fit_transform(X)
            mpd = pt.get_mean_pairwise_euc_distance(pca_X, k=k_eval)
            mpd_null = []
            for j in range(iter2):
                df_np_i_j = pt.get_random_matrix(df_np_i)
                np.seterr(divide='ignore')
                df_np_i_j_delta = cd.likelihood_matrix_array(df_np_i_j, gene_names_i, 'Tenaillon_et_al').get_likelihood_matrix()
                X_j = df_np_i_j_delta/df_np_i_j_delta.sum(axis=1)[:,None]
                X_j -= np.mean(X_j, axis = 0)
                pca_X_j = pca.fit_transform(X_j)
                mpd_null.append(pt.get_mean_pairwise_euc_distance(pca_X_j, k=k_eval))
            p_values.append(len( [m for m in mpd_null if m > mpd] ) / len(mpd_null))
            #z_scores.append( (euc_dist - np.mean(euc_dists)) / np.std(euc_dists) )s
        print(p_values)

        power = len([n for n in p_values if n < 0.05]) / len(p_values)
        print(p_values)
        power_bootstrap = []
        for p in range(iter_bs):
            p_values_sample = random.sample(p_values, sample_bs)
            power_sample = len([n for n in p_values_sample if n < 0.05]) / len(p_values_sample)
            power_bootstrap.append(power_sample)
        power_bootstrap.sort()
        # return number of genes, power, power lower, power upper
        #return  power, power_bootstrap[int(10000*0.025)], power_bootstrap[int(10000*0.975)]
        df_out.write('\t'.join([str(N), str(np.mean(G_list)), str(power), str(power_bootstrap[int(iter_bs*0.025)]), str(power_bootstrap[int(iter_bs*0.975)])]) + '\n')

    df_out.close()



def gene_svd_tenaillon(iter=10000):
    df_path = mydir + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    df_np_delta = cd.likelihood_matrix_array(df_np, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()
    X = df_np_delta/df_np_delta.sum(axis=1)[:,None]
    X -= np.mean(X, axis = 0)
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
        X_i = df_np_delta/df_np_delta.sum(axis=1)[:,None]
        X_i -= np.mean(X_i, axis = 0)
        U_i, s_i, V_i_T = svds(X_i, k=3)
        F_i = (V_i_T.T @ np.diag(s_i)) / np.sqrt(  X_i.shape[0] - 1 )
        vars_null_list.append(np.linalg.norm(F_i, axis=1) ** 2)

    vars_null = np.stack(vars_null_list)
    vars_null_mean = np.mean(vars_null, axis=0)
    vars_null_std = np.std(vars_null, axis=0)
    z_scores = (vars - vars_null_mean) / vars_null_std
    p_values = []
    # calculate p values
    for k, column in enumerate(vars_null.T):
        column_greater = [x for x in column if x < vars[k]]
        p_values.append(len(column_greater) / iter)

    label_z_scores = list(zip(gene_names, z_scores, p_values))
    label_sig_z_scores = [x for x in label_z_scores if x[2] < 0.05]

    print(label_sig_z_scores)

    df_out = open(mydir + '/data/Tenaillon_et_al/gene_z_scores.txt', 'w')
    df_out.write('\t'.join(['Gene', 'z_score', 'p_score']) + '\n')
    for label_z_score in label_z_scores:
        df_out.write('\t'.join([str(label_z_score[0]), str(label_z_score[1]), str(label_z_score[2])]) + '\n')
    df_out.close()



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






#time_partition_ltee()

gene_svd_tenaillon()
