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
        #df_out = open(pt.get_path() + '/data/Good_et_al/permute.txt', 'w')
        #df_out.write('\t'.join(['Iteration','Time', 'MCD']) + '\n')
        # exclude p5 since it doesn't move in ordination space
        to_exclude = pt.complete_nonmutator_lines()
        to_exclude.append('p5')
        df_nonmut = df[df.index.str.contains('|'.join( to_exclude))]
        time_points = [ int(x.split('_')[1]) for x in df_nonmut.index.values]
        time_points_set = sorted(list(set([ int(x.split('_')[1]) for x in df_nonmut.index.values])))
        df_nonmut_array = df_nonmut.as_matrix()
        time_points_positions = {}
        for x in time_points_set:
            time_points_positions[x] =  [i for i,j in enumerate(time_points) if j == x]
        df_0 = df_nonmut.iloc[time_points_positions[time_points_set[0]]]
        for i in range(iter):
            print(i)
            df_0_rndm = pt.random_matrix(df_0.as_matrix())
            df_0_rndm_df =  pd.DataFrame(data= df_0_rndm, index=df_0.index, columns=df_0.columns)
            for tp in time_points_set[1:]:
                df_tp = df_nonmut.iloc[time_points_positions[tp]]
                #tp_minus_df = df_nonmut.iloc[time_points_positions[time_points_set[time_points_set.index(tp) - 1]]]
                df_tp_diff = df_tp.as_matrix() - df_0_rndm
                print(np.sum(df_tp_diff, axis =1))
                # expected combinations of mutations at the next time point
                df_tp_rndm = df_0_rndm + pt.random_matrix(df_tp_diff)
                df_tp_rndm_df = pd.DataFrame(data= df_tp_rndm, index=df_tp.index, columns=df_tp.columns)
                df_0_rndm_df = pd.concat([df_0_rndm_df, df_tp_rndm_df])

            df_0_rndm_df_delta = pt.likelihood_matrix(df_0_rndm_df, 'Good_et_al').get_likelihood_matrix()
            df_0_rndm_df_delta_bc = np.sqrt(pt.get_scipy_bray_curtis(df_0_rndm_df_delta.as_matrix()))
            df_0_rndm_df_delta_cmd = pt.cmdscale(df_0_rndm_df_delta_bc)[0]
            df_0_rndm_df_delta_cmd_df = pd.DataFrame(data= df_0_rndm_df_delta_cmd, index=df_0_rndm_df.index)
            for tp in time_points_set:
                cmd_tp = df_0_rndm_df_delta_cmd_df[df_0_rndm_df_delta_cmd_df.index.str.contains(str(tp))]
                cmd_tp_mcd = pt.get_mean_centroid_distance(cmd_tp.as_matrix(), k = 5)
                #df_out.write('\t'.join([str(i), str(tp), str(cmd_tp_mcd)]) + '\n')

        #df_out.close()


run_permutations(1, dataset = 'good')
#time_permute()
