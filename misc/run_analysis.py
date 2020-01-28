from __future__ import division
import os, re
from random import shuffle
from collections import Counter
import numpy as np
import pandas as pd
import parevol_tools as pt
from sklearn.decomposition import PCA
import functools
from sklearn.metrics.pairwise import euclidean_distances



def run_pca_permutation(iter = 10000, analysis = 'PCA', dataset = 'tenaillon'):
    if dataset == 'tenaillon':
        k = 3
        df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        df_array = df.as_matrix()
        df_out = open(pt.get_path() + '/data/Tenaillon_et_al/permute_' + analysis + '.txt', 'w')
        column_headers = ['Iteration', 'MCD', 'mean_angle', 'mean_dist', 'delta_L', 'x_stat']
        df_out.write('\t'.join(column_headers) + '\n')
        for i in range(iter):
            print(i)
            df_rndm = pd.DataFrame(data=pt.random_matrix(df_array), index=df.index, columns=df.columns)
            df_rndm_delta = pt.likelihood_matrix(df_rndm, 'Tenaillon_et_al').get_likelihood_matrix()
            if analysis == 'PCA':
                X = pt.hellinger_transform(df_rndm_delta)
                pca = PCA()
                df_rndm_delta_out = pca.fit_transform(X)
                #df_pca = pd.DataFrame(data=X_pca, index=df.index)
            mean_angle = pt.get_mean_angle(df_rndm_delta_out, k = k)
            mcd = pt.get_mean_centroid_distance(df_rndm_delta_out, k=k)
            mean_length = pt.get_euc_magnitude_diff(df_rndm_delta_out, k=k)
            mean_dist = pt.get_mean_pairwise_euc_distance(df_rndm_delta_out, k=k)
            x_stat = pt.get_x_stat(pca.explained_variance_[:-1])
            df_out.write('\t'.join([str(i), str(mcd), str(mean_angle), str(mean_dist), str(mean_length), str(x_stat)]) + '\n')
        df_out.close()


    elif dataset == 'good':
        k = 5
        df_path = pt.get_path() + '/data/Good_et_al/gene_by_pop.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        to_exclude = pt.complete_nonmutator_lines()
        to_exclude.append('p5')
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
        column_headers = ['Iteration', 'Generation', 'MCD', 'mean_angle', 'delta_L', 'mean_dist']
        df_out.write('\t'.join(column_headers) + '\n')
        for i in range(iter):
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


def run_pca_sample_size_permutation(iter = 10000, analysis = 'PCA', k =3):
    df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_array = df.as_matrix()
    sample_sizes = np.linspace(2, df.shape[0], num = 20, dtype = int)
    df_out = open(pt.get_path() + '/data/Tenaillon_et_al/sample_size_permute_' + analysis + '.txt', 'w')
    column_headers = ['Sample_size', 'Iteration', 'MCD', 'mean_angle', 'delta_L']
    df_out.write('\t'.join(column_headers) + '\n')
    for sample_size in sample_sizes:
        print("Sample size = "  + str(sample_size))
        for i in range(iter):
            print("Sample size = "  + str(sample_size) + ' Iteration = ' + str(i))
            df_sample = df.sample(n = sample_size)
            #df_sample = df_sample.loc[:, (df_sample != 0).any(axis=0)]
            df_sample_delta = pt.likelihood_matrix(df_sample, 'Tenaillon_et_al').get_likelihood_matrix()
            df_sample_delta = df_sample_delta.loc[:, (df_sample_delta != 0).any(axis=0)]
            X = pt.hellinger_transform(df_sample_delta)
            pca = PCA()
            df_sample_delta_out = pca.fit_transform(X)
            mcd = pt.get_mean_centroid_distance(df_sample_delta_out, k=k)
            mean_angle = pt.get_mean_angle(df_sample_delta_out, k = k)
            mean_length = pt.get_euclidean_distance(df_sample_delta_out, k=k)

            df_out.write('\t'.join([str(sample_size), str(i), str(mcd), str(mean_angle), str(mean_length)]) + '\n')

    df_out.close()


def get_F_stat_pairwise(pca_array, groups, k=3):
    # assuming only two groups
    # groups is a nested list, each list containing row integers for a given group
    X = pca_array[:,0:k]
    between_var = 0
    within_var = 0
    K = len(groups)
    N = np.shape(X)[0]
    euc_sq_dists = np.square(euclidean_distances(X, X))
    SS_T = sum( np.sum(euc_sq_dists, axis =1)) /  (N*2)
    euc_sq_dists_group1 = euc_sq_dists[groups[0][:, None], groups[0]]
    SS_W = sum( np.sum(euc_sq_dists_group1, axis =1)) /  (len(groups[0])*2)
    # between groups sum-of-squares
    SS_A = SS_T - SS_W


    return (SS_A / (K-1)) / (SS_W / (N-K) )




def get_F_stat_mcd(pca_array, groups, k=3):
    X = pca_array[:,0:k]
    mcd_overall = pt.get_mean_centroid_distance(X)
    centroid_overall = np.mean(X, axis = 0)
    between_var = 0
    within_var = 0
    K = len(groups)
    N = np.shape(X)[0]


    for group in groups:
        #euc_dists_group = euc_dists[group[:, None], group]
        X_group = X[group, :]
        mcd_group = pt.get_mean_centroid_distance(X_group)
        groups_values = []
        centroid_group = np.mean(X_group, axis = 0)
        #between_var += ((mcd_group - mcd_group) ** 2) * len(groups)
        between_var += ((centroid_group - centroid_overall) ** 2) * len(groups)


        #centroid_distances = np.sqrt(np.sum(np.square(X_group - np.mean(X_group, axis = 0)), axis=1))
        #within_var += sum( (centroid_distances - mcd_group) ** 2 )

        #within_var += sum(np.sqrt(np.sum(np.square(X_group - np.mean(X_group, axis = 0)), axis=1)))
        within_var += sum(np.sqrt(np.sum(np.square(X_group - centroid_group), axis=1)))

    return (between_var / (K-1)) / (within_var / (N-K) )






def two_treats_sim(iter1 = 1000, iter2 = 1000, alpha = 0.05):
    genes = 10
    pops1 = pops2 = 10
    shape = 1
    scale = 1
    muts1 = muts2 = 20
    to_reshuffle = [0, 5, 10, 15, 20]

    for reshuf in to_reshuffle:
        p_vales = []
        for i in range(iter1):
            #print(i)
            rates = np.random.gamma(shape, scale=scale, size=genes)
            rates1 = rates.copy()
            # permute rates
            shuffle(rates[:reshuf])
            rates2 = rates.copy()
            list_dicts1 = [Counter(np.random.choice(genes, size = muts1, replace = True, p = rates1 / sum(rates1))) for i in range(pops1) ]
            list_dicts2 = [Counter(np.random.choice(genes, size = muts2, replace = True, p = rates2 / sum(rates2))) for i in range(pops2) ]

            df1 = pd.DataFrame(list_dicts1)
            df2 = pd.DataFrame(list_dicts2)
            df = pd.concat([df1, df2])
            df = df.fillna(0)
            count_matrix = df.values
            groups = [ np.asarray(list(range(0, pops1))), np.asarray(list(range(pops1, pops1+pops2))) ]
            pca = PCA()
            X = pt.hellinger_transform(count_matrix)
            pca_fit = pca.fit_transform(X)
            F = get_F_stat_pairwise(pca_fit, groups)
            F_list = []
            for j in range(iter2):
                count_matrix_n0 = pt.random_matrix(count_matrix)
                X_n0 = pt.hellinger_transform(count_matrix_n0)
                pca_fit_n0 = pca.fit_transform(X_n0)
                F_list.append(get_F_stat_pairwise(pca_fit_n0, groups))

            p_vales.append((len([x for x in F_list if x > F]) +1)  / (iter2+1))

        power = (len([k for k in p_vales if k < alpha]) +1)  / (iter1+1)
        print('Reshuffle = ' + str(reshuf) + ', Power ' +  str(power))


    #fig = plt.figure()
    #plt.hist(F_list, bins=50,  alpha=0.8, color = '#175ac6')
    #plt.axvline(F, color = 'red', lw = 3, ls = '--')
    #plt.tight_layout()
    #fig_name = pt.get_path() + '/figs/test_F.png'
    #fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    #plt.close()




    #print(np.mean(F_list))



#two_treats_sim()
#test_stats()
#run_pca_permutation()
#get_likelihood_matrices()
#run_pca_permutation(dataset='tenaillon', iter =10000)
