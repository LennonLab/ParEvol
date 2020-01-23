from __future__ import division
import os, pickle, operator
import random, copy
from itertools import compress
import numpy as np
import pandas as pd
import parevol_tools as pt
import clean_data as cd
import networkx as nx

#import multiprocessing as mp
#from functools import partial
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA


#mydir = '/N/dc2/projects/Lennon_Sequences/ParEvol'
mydir = os.path.expanduser("~/GitHub/ParEvol")


def get_sig_mult_genes(gene_parallelism_statistics, nmin=2):
    # Give each gene a p-value, get distribution
    gene_logpvalues = pt.calculate_parallelism_logpvalues(gene_parallelism_statistics)
    #print(gene_logpvalues)
    pooled_pvalues = []
    for gene_name in gene_logpvalues.keys():
        if (gene_parallelism_statistics[gene_name]['observed']>= nmin) and (float(gene_logpvalues[gene_name]) >= 0):
            pooled_pvalues.append( gene_logpvalues[gene_name] )

    pooled_pvalues = np.array(pooled_pvalues)
    pooled_pvalues.sort()
    #if len(pooled_pvalues) == 0:
    #    continue
    null_pvalue_survival = pt.NullGeneLogpSurvivalFunction.from_parallelism_statistics( gene_parallelism_statistics, nmin=nmin)
    observed_ps, observed_pvalue_survival = pt.calculate_unnormalized_survival_from_vector(pooled_pvalues, min_x=-4)
    # Pvalue version
    # remove negative minus log p values.
    neg_p_idx = np.where(observed_ps>=0)
    observed_ps_copy = observed_ps[neg_p_idx]
    observed_pvalue_survival_copy = observed_pvalue_survival[neg_p_idx]
    pvalue_pass_threshold = np.nonzero(null_pvalue_survival(observed_ps_copy)*1.0/observed_pvalue_survival_copy<pt.get_alpha())[0]

    if len(pvalue_pass_threshold) == 0:
        return []
    else:
        threshold_idx = pvalue_pass_threshold[0]
        pstar = observed_ps_copy[threshold_idx] # lowest value where this is true
        num_significant = observed_pvalue_survival[threshold_idx]

        list_genes = []

        for gene_name in sorted(gene_parallelism_statistics, key=lambda x: gene_parallelism_statistics.get(x)['observed'],reverse=True):
            if gene_logpvalues[gene_name] >= pstar and gene_parallelism_statistics[gene_name]['observed']>=nmin:
                #print(gene_name, gene_parallelism_statistics[gene_name])
                list_genes.append(gene_name)

        return list_genes



# REL606 GenBank accession no. CP000819
def sample_multiplicity_tenaillon(iter1=10000, iter2=10000, bs_size=100):
    df_path = mydir + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    n_rows = df_np.shape[0]
    with open(mydir + '/data/Tenaillon_et_al/gene_size_dict.txt', 'rb') as handle:
        length_dict = pickle.loads(handle.read())
    # get parallelism statistics
    gene_parallelism_statistics = {}
    for gene_i, length_i in length_dict.items():
        gene_parallelism_statistics[gene_i] = {}
        gene_parallelism_statistics[gene_i]['length'] = length_i
        gene_parallelism_statistics[gene_i]['observed'] = 0
        gene_parallelism_statistics[gene_i]['multiplicity'] = 0

    gene_parallelism_statistics_all = copy.deepcopy(gene_parallelism_statistics)
    gene_mut_counts = df.sum(axis=0)
    # save number of mutations for multiplicity
    for locus_tag_i, n_muts_i in gene_mut_counts.iteritems():
        gene_parallelism_statistics_all[locus_tag_i]['observed'] = n_muts_i

    L_mean = np.mean(list(length_dict.values()))
    L_tot = sum(list(length_dict.values()))
    n_tot = sum(gene_mut_counts.values)
    # go back over and calculate multiplicity
    for locus_tag_i in gene_parallelism_statistics_all.keys():
        # double check the measurements from this
        gene_parallelism_statistics_all[locus_tag_i]['multiplicity'] = gene_parallelism_statistics_all[locus_tag_i]['observed'] *1.0/ length_dict[locus_tag_i] * L_mean
        gene_parallelism_statistics_all[locus_tag_i]['expected'] = n_tot*gene_parallelism_statistics_all[locus_tag_i]['length']/L_tot

    sig_genes_all = get_sig_mult_genes(gene_parallelism_statistics_all)
    num_sig_genes_all = len(sig_genes_all)
    print(str(num_sig_genes_all) + ' significant genes w/ all populations')

    df_out=open(mydir + '/data/Tenaillon_et_al/sig_genes_sim.txt', 'w')
    df_out.write('\t'.join(['N', 'genes_mean', 'genes_mean_ci_025', 'genes_mean_ci_975', \
                            'n_mut_mean', 'n_mut_ci_025', 'n_mut_ci_975', \
                            'ESCRE1901_mean', 'ESCRE1901_ci_025', 'ESCRE1901_ci_975',
                            'ECB_01992_mean', 'ECB_01992_ci_025', 'ECB_01992_ci_975']) + '\n')

    Ns = list(range(4, n_rows, 2))
    #Ns = [20]
    for N in Ns:
        print(N)
        gene_num_list = []
        num_sig_genes_list = []
        ESCRE1901_list = []
        ECB_01992_list = []
        n_tot_i_list = []
        for i in range(iter1):
            df_i = df.sample(n = N, replace=False)
            df_i = df_i.loc[:, (df_i != 0).any(axis=0)]
            gene_num_list.append(len(df_i.columns.values))
            gene_parallelism_statistics_i = copy.deepcopy(gene_parallelism_statistics)
            gene_mut_counts_i = df_i.sum(axis=0)
            # save number of mutations for multiplicity
            for locus_tag_i, n_muts_i in gene_mut_counts_i.iteritems():
                gene_parallelism_statistics_i[locus_tag_i]['observed'] = n_muts_i

            n_tot_i = sum(gene_mut_counts_i.values)
            n_tot_i_list.append(n_tot_i)
            # go back over and calculate multiplicity
            for locus_tag_i in gene_parallelism_statistics_i.keys():
                # double check the measurements from this
                gene_parallelism_statistics_i[locus_tag_i]['multiplicity'] = gene_parallelism_statistics_i[locus_tag_i]['observed'] *1.0/ length_dict[locus_tag_i] * L_mean
                gene_parallelism_statistics_i[locus_tag_i]['expected'] = n_tot_i*gene_parallelism_statistics_i[locus_tag_i]['length']/L_tot
            genes_i = get_sig_mult_genes(gene_parallelism_statistics_i)
            num_sig_genes_list.append(len(genes_i))

            # ESCRE1901
            if 'ESCRE1901' in genes_i:
                ESCRE1901_list.append(True)
            else:
                ESCRE1901_list.append(False)

            if 'ECB_01992' in genes_i:
                ECB_01992_list.append(True)
            else:
                ECB_01992_list.append(False)




        num_sig_genes_bs_list = np.sort([np.mean(np.random.choice(num_sig_genes_list, size=bs_size)) for x in range(iter2)])
        num_sig_genes_ci_025 = num_sig_genes_bs_list[int(0.025*iter2)]
        num_sig_genes_ci_975 = num_sig_genes_bs_list[int(0.975*iter2)]
        num_sig_genes_mean = np.mean(num_sig_genes_list)

        n_i_bs_list = np.sort([np.mean(np.random.choice(n_tot_i_list, size=bs_size, replace=True)) for x in range(iter2)])
        n_i_ci_025 = n_i_bs_list[int(0.025*iter2)]
        n_i_ci_975 = n_i_bs_list[int(0.975*iter2)]
        n_i_mean = np.mean(n_tot_i_list)

        ESCRE1901_bs_list = np.sort([sum(np.random.choice(ESCRE1901_list, size=bs_size, replace=True))  / bs_size for x in range(iter2)])
        ESCRE1901_ci_025 = ESCRE1901_bs_list[int(0.025*iter2)]
        ESCRE1901_ci_975 = ESCRE1901_bs_list[int(0.975*iter2)]
        ESCRE1901_mean = sum(ESCRE1901_list) / len(ESCRE1901_list)

        ECB_01992_list_bs_list = np.sort([sum(np.random.choice(ECB_01992_list, size=bs_size, replace=True))  / bs_size for x in range(iter2)])
        ECB_01992_ci_025 = ECB_01992_list_bs_list[int(0.025*iter2)]
        ECB_01992_ci_975 = ECB_01992_list_bs_list[int(0.975*iter2)]
        ECB_01992_mean = sum(ECB_01992_list) / len(ECB_01992_list)

        # get mean proportion

        df_out.write('\t'.join([str(N), str(num_sig_genes_mean), str(num_sig_genes_ci_025), str(num_sig_genes_ci_975), \
                                str(n_i_mean), str(n_i_ci_025), str(n_i_ci_975), \
                                str(ESCRE1901_mean), str(ESCRE1901_ci_025), str(ESCRE1901_ci_975), \
                                str(ECB_01992_mean), str(ECB_01992_ci_025), str(ECB_01992_ci_975)
                                ]) + '\n')

    df_out.close()



def get_bootstrap_power_ci(p_list, power_iter=10000, power_n=50):
    power_sample = []
    for power_iter_i in range(power_iter):
        power_sample.append( len([n for n in np.random.choice(p_list, size=power_n, replace=True) if n < 0.05]) / power_n )
    power_sample.sort()
    power_975 = power_sample[ int(0.975 * power_iter) ]
    power_025 = power_sample[ int(0.025 * power_iter) ]
    return power_025, power_975


def run_ba_ntwk_cov_sims(iter1=1000, iter2=1000):
    df_out = open(mydir + '/data/simulations/cov_ba_ntwrk_methods.txt', 'w')
    df_out.write('\t'.join(['Cov', 'Method', 'Power', 'Power_025', 'Power_975']) + '\n')

    n_pops=100
    n_genes=50
    covs = [0.05, 0.1, 0.15, 0.2]
    #covs = [0.2]
    for cov in covs:
        eig_p_list = []
        mcd_k1_p_list = []
        mcd_k3_p_list = []
        mpd_k1_p_list = []
        mpd_k3_p_list = []
        for i in range(iter1):
            if i %100 ==0:
                print(cov, i)
            lambda_genes = np.random.gamma(shape=3, scale=1, size=n_genes)
            C = pt.get_ba_cov_matrix(n_genes, cov=cov)
            test_cov = np.stack( [pt.get_count_pop(lambda_genes, cov= C) for x in range(n_pops)] , axis=0 )
            X = test_cov/test_cov.sum(axis=1)[:,None]
            X -= np.mean(X, axis = 0)
            pca = PCA()
            pca_fit = pca.fit_transform(X)
            mpd_k1 = pt.get_mean_pairwise_euc_distance(pca_fit,k=1)
            mpd_k3 = pt.get_mean_pairwise_euc_distance(pca_fit,k=3)

            eig = pt.get_x_stat(pca.explained_variance_[:-1], n_features=n_genes)
            mcd_k1 = pt.get_mean_centroid_distance(pca_fit, k = 1)
            mcd_k3 = pt.get_mean_centroid_distance(pca_fit, k = 3)

            #print(pca.explained_variance_[:-1])
            #print(pt.get_x_stat(pca.explained_variance_[:-1]))
            eig_null_list = []
            mcd_k1_null_list = []
            mcd_k3_null_list = []
            mpd_k1_null_list = []
            mpd_k3_null_list = []
            for j in range(iter2):
                test_cov_rndm = pt.get_random_matrix(test_cov)
                X_j = test_cov_rndm/test_cov_rndm.sum(axis=1)[:,None]
                X_j -= np.mean(X_j, axis = 0)
                pca_j = PCA()
                pca_fit_j = pca_j.fit_transform(X_j)
                #pca_fit_j = pca.fit_transform(X_j)
                mpd_k1_null_list.append( pt.get_mean_pairwise_euc_distance(pca_fit_j, k = 1 ) )
                mpd_k3_null_list.append( pt.get_mean_pairwise_euc_distance(pca_fit_j, k = 3 ) )
                mcd_k1_null_list.append(pt.get_mean_centroid_distance(pca_fit_j, k = 1))
                mcd_k3_null_list.append(pt.get_mean_centroid_distance(pca_fit_j, k = 3))
                eig_null_list.append( pt.get_x_stat(pca_j.explained_variance_[:-1], n_features=n_genes) )

            eig_p_list.append(len( [k for k in eig_null_list if k > eig] ) / iter1)
            mcd_k1_p_list.append( len( [k for k in mcd_k1_null_list if k > mcd_k1] ) / iter1 )
            mcd_k3_p_list.append( len( [k for k in mcd_k3_null_list if k > mcd_k3] ) / iter1 )

            mpd_k1_p_list.append( len( [k for k in mpd_k1_null_list if k > mpd_k1] ) / iter1 )
            mpd_k3_p_list.append( len( [k for k in mpd_k3_null_list if k > mpd_k3] ) / iter1 )


        # calculate power
        eig_power = len([n for n in eig_p_list if n < 0.05]) / iter1
        eig_power_025, eig_power_975 = get_bootstrap_power_ci(eig_p_list)

        mcd_k1_power = len([n for n in mcd_k1_p_list if n < 0.05]) / iter1
        mcd_k1_power_025, mcd_k1_power_975 = get_bootstrap_power_ci(mcd_k1_p_list)

        mcd_k3_power = len([n for n in mcd_k3_p_list if n < 0.05]) / iter1
        mcd_k3_power_025, mcd_k3_power_975 = get_bootstrap_power_ci(mcd_k3_p_list)

        mpd_k1_power = len([n for n in mpd_k1_p_list if n < 0.05]) / iter1
        mpd_k1_power_025, mpd_k1_power_975 = get_bootstrap_power_ci(mpd_k1_p_list)

        mpd_k3_power = len([n for n in mpd_k3_p_list if n < 0.05]) / iter1
        mpd_k3_power_025, mpd_k3_power_975 = get_bootstrap_power_ci(mpd_k3_p_list)


        df_out.write('\t'.join([str(cov), 'Eig', str(eig_power), str(eig_power_025), str(eig_power_975)]) + '\n')
        df_out.write('\t'.join([str(cov), 'MCD_k1', str(mcd_k1_power), str(mcd_k1_power_025), str(mcd_k1_power_975)]) + '\n')
        df_out.write('\t'.join([str(cov), 'MCD_k3', str(mcd_k3_power), str(mcd_k3_power_025), str(mcd_k3_power_975)]) + '\n')
        df_out.write('\t'.join([str(cov), 'MPD_k1', str(mpd_k1_power), str(mpd_k1_power_025), str(mpd_k1_power_975)]) + '\n')
        df_out.write('\t'.join([str(cov), 'MPD_k3', str(mpd_k3_power), str(mpd_k3_power_025), str(mpd_k3_power_975)]) + '\n')

    df_out.close()




def run_ba_ntwk_cluster_sims(iter1=1000, iter2=1000, cov=0.2):
    df_out = open(mydir + '/data/simulations/cov_ba_ntwrk_cluster_methods.txt', 'w')
    df_out.write('\t'.join(['Prob', 'CC_mean', 'CC_025', 'CC_975', 'Method', 'Power', 'Power_025', 'Power_975']) + '\n')

    n_pops=100
    n_genes=50
    #covs = [0.05, 0.1, 0.15, 0.2]
    ps = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for p in ps:
        eig_p_list = []
        mcd_k1_p_list = []
        mcd_k3_p_list = []
        mpd_k1_p_list = []
        mpd_k3_p_list = []
        cc_list = []
        for i in range(iter1):
            if i %100 ==0:
                print(cov, i)
            lambda_genes = np.random.gamma(shape=3, scale=1, size=n_genes)
            C, cc = pt.get_ba_cov_matrix(n_genes, cov=cov,  p=p)
            test_cov = np.stack( [pt.get_count_pop(lambda_genes, cov= C) for x in range(n_pops)] , axis=0 )
            X = test_cov/test_cov.sum(axis=1)[:,None]
            X -= np.mean(X, axis = 0)
            pca = PCA()
            pca_fit = pca.fit_transform(X)
            mpd_k1 = pt.get_mean_pairwise_euc_distance(pca_fit,k=1)
            mpd_k3 = pt.get_mean_pairwise_euc_distance(pca_fit,k=3)

            eig = pt.get_x_stat(pca.explained_variance_[:-1], n_features=n_genes)
            mcd_k1 = pt.get_mean_centroid_distance(pca_fit, k = 1)
            mcd_k3 = pt.get_mean_centroid_distance(pca_fit, k = 3)

            eig_null_list = []
            mcd_k1_null_list = []
            mcd_k3_null_list = []
            mpd_k1_null_list = []
            mpd_k3_null_list = []
            for j in range(iter2):
                test_cov_rndm = pt.get_random_matrix(test_cov)
                X_j = test_cov_rndm/test_cov_rndm.sum(axis=1)[:,None]
                X_j -= np.mean(X_j, axis = 0)
                pca_j = PCA()
                pca_fit_j = pca_j.fit_transform(X_j)
                #pca_fit_j = pca.fit_transform(X_j)
                mpd_k1_null_list.append( pt.get_mean_pairwise_euc_distance(pca_fit_j, k = 1 ) )
                mpd_k3_null_list.append( pt.get_mean_pairwise_euc_distance(pca_fit_j, k = 3 ) )
                mcd_k1_null_list.append(pt.get_mean_centroid_distance(pca_fit_j, k = 1))
                mcd_k3_null_list.append(pt.get_mean_centroid_distance(pca_fit_j, k = 3))
                eig_null_list.append( pt.get_x_stat(pca_j.explained_variance_[:-1], n_features=n_genes) )

            #print(len( [k for k in eig_null_list if k > eig] ) / iter1)
            eig_p_list.append(len( [k for k in eig_null_list if k > eig] ) / iter1)
            mcd_k1_p_list.append( len( [k for k in mcd_k1_null_list if k > mcd_k1] ) / iter1 )
            mcd_k3_p_list.append( len( [k for k in mcd_k3_null_list if k > mcd_k3] ) / iter1 )

            mpd_k1_p_list.append( len( [k for k in mpd_k1_null_list if k > mpd_k1] ) / iter1 )
            mpd_k3_p_list.append( len( [k for k in mpd_k3_null_list if k > mpd_k3] ) / iter1 )

            cc_list.append(cc)


        # calculate
        cc_mean = np.mean(cc_list)
        cc_bs_mean_list = []
        for iter_i in range(10000):
            cc_bs_mean_list.append( np.mean( np.random.choice(cc_list, size=50, replace=True ) ))
        cc_bs_mean_list.sort()
        cc_975 = cc_bs_mean_list[ int(0.975 * 10000) ]
        cc_025 = cc_bs_mean_list[ int(0.025 * 10000) ]


        eig_power = len([n for n in eig_p_list if n < 0.05]) / iter1
        eig_power_025, eig_power_975 = get_bootstrap_power_ci(eig_p_list)

        mcd_k1_power = len([n for n in mcd_k1_p_list if n < 0.05]) / iter1
        mcd_k1_power_025, mcd_k1_power_975 = get_bootstrap_power_ci(mcd_k1_p_list)

        mcd_k3_power = len([n for n in mcd_k3_p_list if n < 0.05]) / iter1
        mcd_k3_power_025, mcd_k3_power_975 = get_bootstrap_power_ci(mcd_k3_p_list)

        mpd_k1_power = len([n for n in mpd_k1_p_list if n < 0.05]) / iter1
        mpd_k1_power_025, mpd_k1_power_975 = get_bootstrap_power_ci(mpd_k1_p_list)

        mpd_k3_power = len([n for n in mpd_k3_p_list if n < 0.05]) / iter1
        mpd_k3_power_025, mpd_k3_power_975 = get_bootstrap_power_ci(mpd_k3_p_list)


        df_out.write('\t'.join([str(p), str(cc_mean), str(cc_025), str(cc_975), 'Eig', str(eig_power), str(eig_power_025), str(eig_power_975)]) + '\n')
        df_out.write('\t'.join([str(p), str(cc_mean), str(cc_025), str(cc_975), 'MCD_k1', str(mcd_k1_power), str(mcd_k1_power_025), str(mcd_k1_power_975)]) + '\n')
        df_out.write('\t'.join([str(p), str(cc_mean), str(cc_025), str(cc_975), 'MCD_k3', str(mcd_k3_power), str(mcd_k3_power_025), str(mcd_k3_power_975)]) + '\n')
        df_out.write('\t'.join([str(p), str(cc_mean), str(cc_025), str(cc_975), 'MPD_k1', str(mpd_k1_power), str(mpd_k1_power_025), str(mpd_k1_power_975)]) + '\n')
        df_out.write('\t'.join([str(p), str(cc_mean), str(cc_025), str(cc_975), 'MPD_k3', str(mpd_k3_power), str(mpd_k3_power_025), str(mpd_k3_power_975)]) + '\n')

    df_out.close()





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
#gene_svd_tenaillon()



#cluster_scaling()


#run_ba_ntwk_cov_sims()
#run_ba_ntwk_cluster_sims()
#sample_multiplicity_tenaillon()
