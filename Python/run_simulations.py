from __future__ import division
import os, pickle, operator
from random import shuffle
from itertools import compress
from collections import Counter
import numpy as np
import pandas as pd
import parevol_tools as pt
from sklearn.decomposition import PCA


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
                    # try without hellinger transform
                    #X = pt.hellinger_transform(test_cov)
                    X = pt.get_mean_center(test_cov)
                    pca = PCA()
                    pca_fit = pca.fit_transform(X)
                    euc_dist = pt.get_mean_pairwise_euc_distance(pca_fit)
                    euc_dists = []
                    for j in range(iter2):
                        #X_j = pt.hellinger_transform(pt.get_random_matrix(test_cov))
                        X_j = pt.get_mean_center(pt.get_random_matrix(test_cov))
                        pca_fit_j = pca.fit_transform(X_j)
                        euc_dists.append( pt.get_mean_pairwise_euc_distance(pca_fit_j) )
                    euc_percent = len( [k for k in euc_dists if k < euc_dist] ) / len(euc_dists)
                    z_score = (euc_dist - np.mean(euc_dists)) / np.std(euc_dists)
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
                #X = pt.hellinger_transform(test_cov)
                X = pt.get_mean_center(test_cov)
                pca = PCA()
                pca_fit = pca.fit_transform(X)
                euc_dist = pt.get_mean_pairwise_euc_distance(pca_fit)
                euc_dists = []
                for j in range(iter2):
                    #X_j = pt.hellinger_transform(pt.get_random_matrix(test_cov))
                    X_j = pt.get_mean_center(pt.get_random_matrix(test_cov))
                    pca_fit_j = pca.fit_transform(X_j)
                    euc_dists.append( pt.get_mean_pairwise_euc_distance(pca_fit_j) )
                euc_percent = len( [k for k in euc_dists if k < euc_dist] ) / len(euc_dists)
                z_score = (euc_dist - np.mean(euc_dists)) / np.std(euc_dists)
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
                X = pt.get_mean_center(test_cov)
                pca = PCA()
                pca_fit = pca.fit_transform(X)
                euc_dist = pt.get_mean_pairwise_euc_distance(pca_fit)
                euc_dists = []
                for j in range(iter2):
                    X_j = pt.get_mean_center(pt.get_random_matrix(test_cov))
                    pca_fit_j = pca.fit_transform(X_j)
                    euc_dists.append( pt.get_mean_pairwise_euc_distance(pca_fit_j) )
                euc_percent = len( [k for k in euc_dists if k < euc_dist] ) / len(euc_dists)
                z_score = (euc_dist - np.mean(euc_dists)) / np.std(euc_dists)
                df_out.write('\t'.join([str(N), str(G), str(cov), str(prop), str(i), str(euc_percent), str(z_score)]) + '\n')
            print(N, G, cov)
    df_out.close()



#def run_ba_cov_rho_sims():
    





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









def run_ba_cor_sub_sims(shape=1, scale=1, G = 50, N = 50):
    cov = 0.1
    lambda_genes = np.random.gamma(shape=shape, scale=scale, size=G)
    lambda_genes.sort()
    C, edge_count = pt.get_ba_cov_matrix(G, cov, get_node_edge_sum=True)
    zipped = list(zip( list(range(G)) , edge_count.tolist()[0] ))
    zipped.sort(key = lambda t: t[1])
    # figure out how to sort covariance matrix
    total_inversions = 30
    inversion_count = 0
    while inversion_count < total_inversions:
        pair = np.random.choice(list(range(G)), size = 2, replace=False)
        pair.sort()
        if lambda_genes[pair[0]] < lambda_genes[pair[1]]:
            lambda_0 =lambda_genes[pair[0]].copy()
            lambda_1 =lambda_genes[pair[1]].copy()
            lambda_genes[pair[0]] = lambda_1
            lambda_genes[pair[1]] = lambda_0
            inversion_count += 1
    unzipped = list(zip(*zipped))
    rezipped = list( zip( unzipped[0], unzipped[1],  lambda_genes) )
    rezipped.sort(key = lambda t: t[0])
    unrezipped = list(zip(*rezipped))
    lambda_genes_sorted = list(unrezipped[2])

    print(lambda_genes_sorted)



    #list[min(lambda_genes[pair[0]],j)] < list[max(i,j)]
    #print()

def rndm_sample_tenaillon(iter1=1000, iter2=1000):
    df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    n_rows = df_np.shape[0]
    df_out=open(pt.get_path() + '/data/Tenaillon_et_al/sample_size_sim.txt', 'w')
    df_out.write('\t'.join(['N', 'G', 'Iteration', 'dist_percent', 'z_score']) + '\n')
    Ns = list(range(2, 40, 2))
    for N in Ns:
        for i in range(iter1):
            #df_np_i = df_np[np.random.choice(n_rows, N, replace=False), :]
            #df_np_i = df_np_i[: , ~np.all(df_np_i == 0, axis=0)]
            #df_i = df.sample(N)
            df_np_i = df_np[np.random.randint(n_rows, size=N), :]
            gene_bool = np.all(df_np_i == 0, axis=0)
            # flip around to select gene_size
            gene_names_i = list(compress(gene_names, list(map(operator.not_, gene_bool))))
            df_np_i = df_np_i[:,~np.all(df_np_i == 0, axis=0)]
            #df_i = df_i.loc[:, (df_i != 0).any(axis=0)]
            np.seterr(divide='ignore')
            df_np_i_delta = pt.likelihood_matrix_array(df_np_i, gene_names_i, 'Tenaillon_et_al').get_likelihood_matrix()
            X = pt.hellinger_transform(df_np_i_delta)
            pca = PCA()
            pca_fit = pca.fit_transform(X)
            euc_dist = pt.get_mean_pairwise_euc_distance(pca_fit)
            euc_dists = []
            for j in range(iter2):
                #df_np_i_j = pt.random_matrix(df_np_i)
                df_np_i_j = pt.get_random_matrix(df_np_i)
                np.seterr(divide='ignore')
                df_np_i_j_delta = pt.likelihood_matrix_array(df_np_i_j, gene_names_i, 'Tenaillon_et_al').get_likelihood_matrix()
                #df_i_j = pd.DataFrame(data=pt.random_matrix(df_np_i_j), index=df_i.index, columns=df_i.columns)
                #df_i_j_delta = pt.likelihood_matrix(df_i_j, 'Tenaillon_et_al').get_likelihood_matrix()
                X_j = pt.hellinger_transform(df_np_i_j_delta)
                pca_fit_j = pca.fit_transform(X_j)
                euc_dists.append( pt.get_mean_pairwise_euc_distance(pca_fit_j) )

            G = df_np_i.shape[1]
            euc_percent = len( [k for k in euc_dists if k < euc_dist] ) / len(euc_dists)
            z_score = (euc_dist - np.mean(euc_dists)) / np.std(euc_dists)
            print(str(N), str(i), str(G), str(euc_percent), str(z_score))
            df_out.write('\t'.join([str(N), str(G), str(i), str(euc_percent), str(z_score)]) + '\n')

    df_out.close()




#rndm_sample_tenaillon()

# write code to re-shuffle proportion of positive/negative correlations

#run_ba_cor_sub_sims()