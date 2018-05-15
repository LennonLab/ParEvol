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
                df_out.write('\t'.join([str(i), str(tp), str(cmd_tp_mcd)]) + '\n')
        #for i in range(iter):
        #    print(i)
        #    df_rndm = pd.DataFrame(data=pt.random_matrix(df_nonmut_array), index=df_nonmut.index, columns=df_nonmut.columns)
        #    df_rndm_delta = pt.likelihood_matrix(df_rndm, 'Good_et_al').get_likelihood_matrix()
        #    df_rndm_delta_bc = np.sqrt(pt.get_scipy_bray_curtis(df_rndm_delta.as_matrix()))
        #    df_rndm_delta_cmd = pt.cmdscale(df_rndm_delta_bc)[0]
        #    for tp in time_points_set:
        #        tp_df_rndm_delta_cmd = df_rndm_delta_cmd[time_points_positions[tp][:, None], :]
        #        tp_mcd = pt.get_mean_centroid_distance(tp_df_rndm_delta_cmd, k = 5)
        #        df_out.write('\t'.join([str(i), str(tp), str(tp_mcd)]) + '\n')
        df_out.close()



def time_series_mc():
    df_path = pt.get_path() + '/data/Good_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    to_exclude = pt.complete_nonmutator_lines()
    to_exclude.append('p5')
    df_nonmut = df[df.index.str.contains('|'.join( to_exclude))]
    time_points_set = sorted(list(set([ int(x.split('_')[1]) for x in df_nonmut.index.values])))
    df_gene = df_nonmut[['hokB']]
    total_mut_dict = {}
    for x in time_points_set:
        total_count = sum(df_gene[df_gene.index.str.contains('_' + str(x))].values)
        print(x, total_count)
        #print(x,df_gene[df_gene.index.str.contains(str(x))].values )#np.array([i for i,j in enumerate(time_points) if j == x])

    #print(sum(df_gene.values))

    total_mut_dict = {}

def time_permute():
    # rows = p1t1, p1t2, p2t1, p2t2
    # columns = gene1, gene2
    test_time_array = np.array([[0, 1], [2, 1], [0, 0], [1, 2]])
    t1 = test_time_array[[0,2],:]
    t2 = test_time_array[[1,3],:]
    r_sum_1 = np.sum(test_time_array[[0,2],:], axis = 1)
    c_sum_1 = np.sum(test_time_array[[0,2],:], axis = 0)
    r_sum_2 = np.sum(test_time_array[[1,3],:], axis = 1)
    c_sum_2 = np.sum(test_time_array[[1,3],:], axis = 0)
    #print(r_sum_2 - r_sum_1)
    print(t1)
    print(t2)
    print(t2 - t1)
    print(pt.random_matrix(t2 - t1))
    #print('row sum T1', np.sum(test_time_array[[0,2],:], axis = 1))
    #print('column sum T1', np.sum(test_time_array[[0,2],:], axis = 0))
    #print('row sum T2', np.sum(test_time_array[[1,3],:], axis = 1))
    #print('column sum T2', np.sum(test_time_array[[1,3],:], axis = 0))


# t1 row sum =


'''
Write new randomization procedure for time series
Constrain row and column sum, but
'''


run_permutations(100, dataset = 'good')
#time_permute()
