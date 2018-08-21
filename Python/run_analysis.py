from __future__ import division
import numpy as np
import pandas as pd
import parevol_tools as pt
from sklearn.decomposition import PCA
import functools

def run_permutation(iter, analysis = 'PCA', dataset = 'tenaillon'):
    if dataset == 'tenaillon':
        df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        df_array = df.as_matrix()
        df_out = open(pt.get_path() + '/data/Tenaillon_et_al/permute_' + analysis + '.txt', 'w')
        column_headers = ['Iteration', 'MCD']
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
            df_out.write('\t'.join([str(i), str(pt.get_mean_centroid_distance(df_rndm_delta_out, k=3))]) + '\n')
        df_out.close()


    elif dataset == 'good':
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
        column_headers = ['Iteration', 'Generation', 'MCD']
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
                df_out.write('\t'.join([str(i), str(tp), str(pt.get_mean_centroid_distance(df_rndm_delta_out_tp.as_matrix(), k=3))]) + '\n')

        df_out.close()


#run_permutation(10000, dataset = 'good')
#time_permute()
