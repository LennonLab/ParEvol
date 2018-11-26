from __future__ import division
import os, re
from collections import Counter
import numpy as np
import pandas as pd
import parevol_tools as pt
from sklearn.decomposition import PCA
import functools

'''PCA code'''


def get_likelihood_matrices():
    df_good_path = pt.get_path() + '/data/Good_et_al/gene_by_pop.txt'
    df_good =  pd.read_csv(df_good_path, sep = '\t', header = 'infer', index_col = 0)
    df_good_delta = pt.likelihood_matrix(df_good, 'Good_et_al').get_likelihood_matrix()
    df_good_delta_out = pt.get_path() + '/data/Good_et_al/gene_by_pop_delta.txt'
    df_good_delta.to_csv(df_good_delta_out, sep = '\t', index = True)

    df_good_poly_path = pt.get_path() + '/data/Good_et_al/gene_by_pop_poly.txt'
    df_good_poly =  pd.read_csv(df_good_poly_path, sep = '\t', header = 'infer', index_col = 0)
    df_good_poly_delta = pt.likelihood_matrix(df_good_poly, 'Good_et_al').get_likelihood_matrix()
    df_good_poly_delta_out = pt.get_path() + '/data/Good_et_al/gene_by_pop_poly_delta.txt'
    df_good_poly_delta.to_csv(df_good_poly_delta_out, sep = '\t', index = True)



def run_pca_permutation(iter = 10000, analysis = 'PCA', dataset = 'tenaillon'):
    if dataset == 'tenaillon':
        k = 3
        df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        df_array = df.as_matrix()
        df_out = open(pt.get_path() + '/data/Tenaillon_et_al/permute_' + analysis + '.txt', 'w')
        column_headers = ['Iteration', 'MCD', 'mean_angle', 'mean_dist', 'delta_L']
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
            elif analysis == 'cMDS':
                df_rndm_delta_bc = np.sqrt(pt.get_bray_curtis(df_rndm_delta.as_matrix()))
                df_rndm_delta_out = pt.cmdscale(df_rndm_delta_bc)[0]
            else:
                continue
            mean_angle = pt.get_mean_angle(df_rndm_delta_out, k = k)
            mcd = pt.get_mean_centroid_distance(df_rndm_delta_out, k=k)
            mean_length = pt.get_euc_magnitude_diff(df_rndm_delta_out, k=k)
            mean_dist = pt.get_mean_pairwise_euc_distance(df_rndm_delta_out, k=k)
            df_out.write('\t'.join([str(i), str(mcd), str(mean_angle), str(mean_dist), str(mean_length)]) + '\n')
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



#get_likelihood_matrices()
#run_pca_permutation(dataset='good', iter =10000)
