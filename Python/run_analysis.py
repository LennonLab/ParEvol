from __future__ import division
import numpy as np
import pandas as pd
import parevol_tools as pt

test_array = np.array([[1, 2], [3, 4], [3, 0]])


def run_permutations(iter, dataset = 'tenaillon'):
    if dataset == 'tenaillon':
        df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        df_array = df.as_matrix()
        df_out = open(pt.get_path() + '/data/Tenaillon_et_al/permute.txt', 'w')
        column_headers = ['Iteration', 'MCD']
        df_out.write('\t'.join(column_headers) + '\n')
        for i in range(iter):
            print(i)
            df_rndm = pd.DataFrame(data=pt.random_matrix(df_array), index=df.index, columns=df.columns)
            df_rndm_delta = pt.likelihood_matrix(df_rndm, 'Tenaillon_et_al').get_likelihood_matrix()
            df_rndm_delta_bc = np.sqrt(pt.get_bray_curtis(df_rndm_delta.as_matrix()))
            df_rndm_delta_cmd = pt.cmdscale(df_rndm_delta_bc)[0]
            df_out.write('\t'.join([str(i), str(pt.get_mean_centroid_distance(df_rndm_delta_cmd, k=3))]) + '\n')
        df_out.close()


    elif dataset == 'good':
        df_path = pt.get_path() + '/data/Good_et_al/gene_by_pop.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        df_out = open(pt.get_path() + '/data/Good_et_al/permute.txt', 'w')
        df_out.write('\t'.join(['Iteration','Time', 'MCD']) + '\n')
        # exclude p5 since it doesn't move in ordination space
        to_exclude = pt.complete_nonmutator_lines()
        to_exclude.append('p5')
        df_nonmut = df[df.index.str.contains('|'.join( to_exclude))]
        time_points = [ int(x.split('_')[1]) for x in df_nonmut.index.values]
        time_points_set = list(set([ int(x.split('_')[1]) for x in df_nonmut.index.values]))
        df_nonmut_array = df_nonmut.as_matrix()
        time_points_positions = {}
        for x in time_points_set:
            time_points_positions[x] =  np.array([i for i,j in enumerate(time_points) if j == x])
        for i in range(iter):
            print(i)
            df_rndm = pd.DataFrame(data=pt.random_matrix(df_nonmut_array), index=df_nonmut.index, columns=df_nonmut.columns)
            df_rndm_delta = pt.likelihood_matrix(df_rndm, 'Good_et_al').get_likelihood_matrix()
            df_rndm_delta_bc = np.sqrt(pt.get_scipy_bray_curtis(df_rndm_delta.as_matrix()))
            df_rndm_delta_cmd = pt.cmdscale(df_rndm_delta_bc)[0]
            for tp in time_points_set:
                tp_df_rndm_delta_cmd = df_rndm_delta_cmd[time_points_positions[tp][:, None], :]
                tp_mcd = pt.get_mean_centroid_distance(tp_df_rndm_delta_cmd, k = 5)
                df_out.write('\t'.join([str(i), str(tp), str(tp_mcd)]) + '\n')
        df_out.close()



run_permutations(10, dataset = 'good')
#plot_pcoa('good')
