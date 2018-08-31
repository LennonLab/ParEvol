from __future__ import division
import os, re
import numpy as np
import pandas as pd
import parevol_tools as pt
from sklearn.decomposition import PCA
import functools

def run_permutation(iter = 10000, analysis = 'PCA', dataset = 'tenaillon'):
    if dataset == 'tenaillon':
        k = 3
        df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        df_array = df.as_matrix()
        df_out = open(pt.get_path() + '/data/Tenaillon_et_al/permute_' + analysis + '.txt', 'w')
        column_headers = ['Iteration', 'MCD', 'mean_angle', 'delta_L']
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
            mean_length = pt.get_euclidean_distance(df_rndm_delta_out, k=k)
            df_out.write('\t'.join([str(i), str(mcd), str(mean_angle), str(mean_length)]) + '\n')
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
        column_headers = ['Iteration', 'Generation', 'MCD', 'mean_angle', 'delta_L']
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
                mean_angle = pt.get_mean_angle(df_rndm_delta_out_tp.as_matrix(), k = k)
                mcd = pt.get_mean_centroid_distance(df_rndm_delta_out_tp.as_matrix(), k=k)
                mean_length = pt.get_euclidean_distance(df_rndm_delta_out_tp.as_matrix(), k=k)
                df_out.write('\t'.join([str(i), str(tp), str(mcd), str(mean_angle), str(mean_length) ]) + '\n')

        df_out.close()


def run_sample_size_permutation(iter = 10000, analysis = 'PCA', k =3):
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




def get_kmax_random_networks(iterations = 1000):
    directory = pt.get_path() + '/data/Good_et_al/networks_BIC/'
    kmax_dict = {}
    df_out = open(pt.get_path() + '/data/Good_et_al/networks_BIC_rndm.txt', 'w')
    df_out.write('\t'.join(['Generations', 'Iteration', 'k_max']) + '\n')
    for filename in os.listdir(directory):
        df = pd.read_csv(directory + filename, sep = '\t', header = 'infer', index_col = 0)
        gens = filename.split('.')
        time = re.split('[_.]', filename)[1]
        iter_list = []
        print(time)
        for iter in range(iterations):
            random_matrix = pt.get_random_network(df)
            # -1 because the sum includes the node interacting with itself
            kmax_iter = int(max(np.sum(random_matrix, axis=0)) - 1)
            #iter_list.append(kmax_iter)
            df_out.write('\t'.join([str(time), str(iter), str(kmax_iter)]) + '\n')
        #kmax_dict[time] = iter_list
    df_out.close()



def get_clustering_coefficients(dataset = 'tenaillon'):
    if dataset == 'tenaillon':
        # df is a numpy matrix or pandas dataframe containing network interactions
        df_path = pt.get_path() + '/data/Tenaillon_et_al/network.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        df_out = open(pt.get_path() + '/data/Tenaillon_et_al/network_CCs.txt', 'w')
        df_out.write('\t'.join(['Gene', 'k_i', 'C_i']) + '\n')
        for index, row in df.iterrows():
            k_row = sum(i != 0 for i in row.values) - 1
            if (k_row == 0) or (k_row == 1):
                C_i = 0
            else:
                non_zero = row.nonzero()
                row_non_zero = row[non_zero[0]]
                # drop the node
                row_non_zero = row_non_zero.drop(labels = [index])
                L_i = 0
                for index_gene, gene in row_non_zero.iteritems():
                    row_non_zero_list = row_non_zero.index.tolist()
                    row_non_zero_list.remove(index_gene)
                    df_subset = df.loc[[index_gene]][row_non_zero_list]
                    L_i += sum(sum(i != 0 for i in df_subset.values))
                # we don't multiply L_i by a factor of 2 bc we're double counting edges
                C_i =  L_i  / (k_row * (k_row-1) )
            df_out.write('\t'.join([index, str(k_row), str(C_i)]) + '\n')
        df_out.close()

    elif dataset == 'good':
        directory = pt.get_path() + '/data/Good_et_al/networks_BIC/'
        df_out = open(pt.get_path() + '/data/Good_et_al/network_CCs.txt', 'w')
        df_out.write('\t'.join(['Generations', 'Gene', 'k_i', 'C_i']) + '\n')
        for filename in os.listdir(directory):
            df = pd.read_csv(directory + filename, sep = '\t', header = 'infer', index_col = 0)
            gens = filename.split('.')
            time = re.split('[_.]', filename)[1]
            print(time)
            for index, row in df.iterrows():
                k_row = sum(i != 0 for i in row.values) - 1
                if (k_row == 0) or (k_row == 1):
                    C_i = float(0)
                else:
                    non_zero = row.nonzero()
                    row_non_zero = row[non_zero[0]]
                    # drop the node
                    row_non_zero = row_non_zero.drop(labels = [index])
                    L_i = 0
                    for index_gene, gene in row_non_zero.iteritems():
                        row_non_zero_list = row_non_zero.index.tolist()
                        row_non_zero_list.remove(index_gene)
                        df_subset = df.loc[[index_gene]][row_non_zero_list]
                        L_i += sum(sum(i != 0 for i in df_subset.values))
                    # we don't multiply L_i by a factor of 2 bc we're double counting edges
                    C_i =  L_i  / (k_row * (k_row-1) )
                df_out.write('\t'.join([str(time), index, str(k_row), str(C_i)]) + '\n')

        df_out.close()




def get_good_network_features():
    directory = pt.get_path() + '/data/Good_et_al/networks_BIC/'
    df_out = open(pt.get_path() + '/data/Good_et_al/network_features.txt', 'w')
    df_out_columns = ['Generations', 'N', 'k_max', 'k_mean', 'C_mean', 'C_mean_no1or2', 'd_mean']
    df_out.write('\t'.join(df_out_columns) + '\n')

    df_clust_path = pt.get_path() + '/data/Good_et_al/network_CCs.txt'
    df_clust = pd.read_csv(df_clust_path, sep = '\t', header = 'infer')#, index_col = 0)
    for filename in os.listdir(directory):
        df = pd.read_csv(directory + filename, sep = '\t', header = 'infer', index_col = 0)
        gens = filename.split('.')
        time = re.split('[_.]', filename)[1]
        df_clust_time = df_clust.loc[df_clust['Generations'] == int(time)]
        N = df.shape[0]
        k_max = max(df_clust_time.k_i.values)
        k_mean = np.mean(df_clust_time.k_i.values)
        C_mean = np.mean(df_clust_time.C_i.values)
        C_mean_no1or2 = np.mean(df_clust_time.loc[df_clust_time['k_i'] >= 2].C_i.values)

        distance_df = pt.networkx_distance(df)
        print(time)
        print(distance_df)

        row = [str(time), str(N), str(k_max), str(k_mean), str(C_mean), str(C_mean_no1or2), str(distance_df)]
        df_out.write('\t'.join(row) + '\n')

    df_out.close()




#get_good_network_features()
#run_permutation(dataset = 'good')
#run_permutation(dataset = 'tenaillon')
#run_sample_size_permutation()
#run_permutation()
#get_good_network_features()
