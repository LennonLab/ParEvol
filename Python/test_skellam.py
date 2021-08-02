from __future__ import division

import sys, pickle, math, random, os, itertools, re, collections
from itertools import combinations
import parevol_tools as pt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd
import numpy as np

import scipy.stats as stats
from scipy.special import iv

import statsmodels.stats.multitest as mt

mydir = os.path.expanduser("~/GitHub/ParEvol")





df_turner_path = mydir + '/data/Turner_et_al/gene_by_pop.txt'
df_turner = pd.read_csv(df_turner_path, sep = '\t', header = 'infer', index_col = 0)


output_filename = mydir + '/data/Turner_et_al/significant_genes_skellam.pickle'


##### add empty genes
# 6,943

cols_to_add = 6943 - df_turner.shape[1]

empty_cols = pd.DataFrame(np.zeros((df_turner.shape[0], cols_to_add)))
#empty_cols.set_index = df_turner.index.values

#empty_cols = empty_cols.set_index(df_turner.index.values)
empty_cols.index = df_turner.index.values


df_turner  = pd.concat([df_turner, empty_cols], axis=1)


#df_turner = df_turner.loc[:, (df_turner != 0).any(axis=0)]

turner_treats = ['high_carbon_large_bead', 'high_carbon_planktonic', 'low_carbon_large_bead', 'low_carbon_planktonic']


def calculate_G(matrix):

    treatment_1_2_np_rel = matrix/matrix.sum(axis=1)[:,None]

    difference = treatment_1_2_np_rel[0,:] - treatment_1_2_np_rel[1,:]
    difference = np.absolute(difference)

    difference = difference[difference>0]

    G = sum(difference * np.log(difference)) * -1

    return G


def calculate_difference(matrix):

    treatment_1_2_np_rel = matrix/matrix.sum(axis=1)[:,None]

    difference = treatment_1_2_np_rel[0,:] - treatment_1_2_np_rel[1,:]
    difference = np.absolute(difference)

    return np.mean(difference)





def skellam_pdf(delta_n, lambda_1, lambda_2):

    if delta_n > 0:

        pmf = ((lambda_1/lambda_2)**(delta_n/2)) * iv(delta_n, 2*np.sqrt(lambda_1*lambda_2))
        pmf += ((lambda_2/lambda_1)**(delta_n/2)) * iv(-1*delta_n, 2*np.sqrt(lambda_1*lambda_2))
        pmf *= np.exp((-1*lambda_1) + (-1*lambda_2))

    else:

        pmf = np.exp((-1*lambda_1) + (-1*lambda_2)) * iv(0, 2*np.sqrt(lambda_1*lambda_2))

    return pmf



def calculate_survival(counts_1, counts_2, genes, n_min = 3, alpha = 0.05):

    #counts_1 = np.asarray([8,3,4,0,0,0,2,2,7,2,4,2,0,2,2,0,2,2,2,0,4,2,0,2,2,2,0,2,2,0,2,2,2,0,2,2,0,2,2,2,0,2,2,0,2,2,0,2,2,2,0,2,2])
    #counts_2 = np.asarray([0,8,2,0,1,0,3,2,0,2,2,2,0,2,2,0,2,2,2,0,2,2,3,2,2,2,0,2,2,0,2,2,2,0,2,2,0,2,2,2,0,2,2,0,2,2,0,2,2,2,0,2,2])

    lambda_1 = sum(counts_1)/len(counts_1)
    lambda_2 = sum(counts_2)/len(counts_2)

    delta_n_original = np.absolute(counts_1-counts_2)
    delta_n = delta_n_original[delta_n_original>n_min]

    genes = genes[delta_n_original>n_min]

    delta_n_range = list(range(0,1000))
    delta_n_range_array = np.asarray(delta_n_range)


    delta_n_no_absolute = counts_1-counts_2
    delta_n_no_absolute = delta_n_no_absolute[delta_n_original>n_min]



    delta_n_range_array_subset = delta_n_range_array[delta_n_range_array<=max(delta_n)]

    pmf = [skellam_pdf(i, lambda_1, lambda_2) for i in delta_n_range]
    pmf = np.asarray(pmf)

    survival_null = [ 1-sum(pmf[:i]) for i in range(len(pmf)) ]
    survival_null = np.asarray(survival_null)
    survival_null = survival_null[delta_n_range_array<=max(delta_n)]

    survival_obs = [ len(delta_n[delta_n>=i])/len(delta_n) for i in delta_n_range]
    survival_obs = np.asarray(survival_obs)
    survival_obs = survival_obs[delta_n_range_array<=max(delta_n)]

    P_values = [sum(pmf[delta_n_range.index(delta_n_i):]) for delta_n_i in delta_n]
    P_values = np.asarray(P_values)

    expected_number_genes = 0
    P_range = np.linspace(10**-4, 0.05, num=10000)[::-1]


    N_bar_P_star_div_N_P_star = []
    P_stars = []

    N_bar_P = []

    for P_range_i in P_range:

        N_P_star = len(P_values[P_values<P_range_i])
        N_bar_P_star = 0

        if N_P_star == 0:
            continue

        #for g in range(len(counts_1)):

        for delta_n_j_idx, delta_n_j in enumerate(delta_n):
            if delta_n_j < n_min:
                continue

            P_delta_n_j = sum(pmf[delta_n_range.index(delta_n_j):])

            if P_range_i > P_delta_n_j:
                # no gene specific indices so just multiply the final probability by number of genes
                N_bar_P_star += skellam_pdf(delta_n_j, lambda_1, lambda_2) * len(delta_n_original)

        #print(P_range_i, N_bar_P_star)
        N_bar_P_star_div_N_P_star.append(N_bar_P_star/N_P_star)
        P_stars.append(P_range_i)

        #N_bar_P.append(N_bar_P_star)

    N_bar_P_star_div_N_P_star = np.asarray(N_bar_P_star_div_N_P_star)
    P_stars = np.asarray(P_stars)

    position_P_star = np.argmax(N_bar_P_star_div_N_P_star<=0.05)

    P_star = P_stars[position_P_star]

    #N_bar_P = np.asarray(N_bar_P)


    return delta_n_range_array_subset, genes, survival_obs, survival_null, delta_n_no_absolute, P_values, P_star





#fig = plt.figure(figsize=(4, 4))
#gs = gridspec.GridSpec(nrows=3, ncols=3)


def get_skellam_significant_genes():

#treatment_combinations = list(itertools.combinations(turner_treats, 2))

    treatment_combinations_dict = {}

    for treatment_i_idx, treatment_i in enumerate(turner_treats):
        for treatment_j_idx, treatment_j in enumerate(turner_treats):

            if treatment_j_idx <= treatment_i_idx:
                continue

            #ax = fig.add_subplot(gs[treatment_i_idx, treatment_j_idx-1])

            print(treatment_i, treatment_j)

            #ax_genes = fig.add_subplot(gs[treatment_j_idx, treatment_i_idx])

            n_min = 2

            #treatment_i, treatment_2 = subset

            treatment_1_df = df_turner.filter(like=treatment_i, axis=0)
            treatment_2_df = df_turner.filter(like=treatment_j, axis=0)

            treatment_1_df_sum = treatment_1_df.sum(axis=0)
            treatment_2_df_sum = treatment_2_df.sum(axis=0)

            treatment_1_2_df = pd.concat([treatment_1_df_sum, treatment_2_df_sum], axis=1)
            treatment_1_2_df = treatment_1_2_df.T

            #treatment_1_2_df = treatment_1_2_df[treatment_1_2_df.columns[treatment_1_2_df.sum()>0]]

            treatment_1_2_np = treatment_1_2_df.values

            treatment_1_np = treatment_1_2_np[0,:]
            treatment_2_np = treatment_1_2_np[1,:]


            # calculate number of genes
            genes = treatment_1_2_df.columns.values


            delta_n_range_array_subset, genes_keep, survival_obs, survival_null, delta_n, P_values, P_star = calculate_survival(treatment_1_np, treatment_2_np, genes, n_min=n_min)

            #delta_n = np.absolute(treatment_1_np-treatment_2_np)

            P_values_significant = P_values[P_values<P_star]
            genes_significant = genes_keep[P_values<P_star]
            delta_n = delta_n[P_values<P_star]



            treatment_combinations_dict[(treatment_i, treatment_j)] = {}
            treatment_combinations_dict[(treatment_i, treatment_j)]['P_values_significant'] = P_values_significant
            treatment_combinations_dict[(treatment_i, treatment_j)]['genes_significant'] = genes_significant
            treatment_combinations_dict[(treatment_i, treatment_j)]['delta_n'] = delta_n


            #G_null_list = []

            #for i in range(10000):

            #    treatment_1_2_np_rndm = pt.get_random_matrix(treatment_1_2_np)

            #    G_i = calculate_difference(treatment_1_2_np_rndm)

            #    G_null_list.append(G_i)

            #G_null_list = np.asarray(treatment_1_2_np_rndm)

            #G = calculate_difference(treatment_1_2_np)
            #P = (len(G_null_list[G_null_list>G]) + 1)/ (10000 + 1)




            #difference = np.absolute(treatment_1_2_np[0,:] - treatment_1_2_np[1,:])

            #difference_sort = np.sort(difference)
            #cdf = 1-  np.arange(len(difference_sort))/float(len(difference_sort))

            #ax.plot(difference_sort, cdf, c='grey', ls='--', alpha=1, zorder=1)

            #ax.plot(difference_sort, cdf, c='dodgerblue', ls='--', alpha=1, zorder=2)

            #ax.plot(delta_n_range_array_subset, survival_null, c='grey', ls='--', alpha=1, zorder=1)
            #ax.plot(delta_n_range_array_subset, survival_obs, c='dodgerblue', ls='--', alpha=1, zorder=2)




            #ax.plot(difference_sort, cdf, c='dodgerblue', ls='--', alpha=0.8)

            #print(len(survival_null), len(delta_n), len(delta_n_range_array_subset))

            #ax.set_xlabel('Mutation difference, ' + r'$\left | \Delta n \right |$', fontsize=6)
            #ax.set_ylabel('Fraction genes, ' + r'$\geq \left | \Delta n \right |$', fontsize=6)


            #ax.hist(difference, bins=10, density=True, alpha=0.7)

            #ax.set_ylim([0.1, 1.1])

            #ax.set_yscale('log', base=10)

            #ax.xaxis.set_tick_params(labelsize=6)
            #ax.yaxis.set_tick_params(labelsize=6)


    with open(output_filename, 'wb') as handle:
            pickle.dump(treatment_combinations_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#fig.tight_layout()
#fig.savefig(mydir + '/figs/difference_absolute.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
#plt.close()



#get_skellam_significant_genes()

with open(output_filename, 'rb') as handle:
    treatment_combinations_dict = pickle.load(handle)


#for treatment_i_idx, treatment_i in enumerate(turner_treats):
#    for treatment_j_idx, treatment_j in enumerate(turner_treats):


genes_enriched_all_dict = {}

for turner_treat in turner_treats:

    genes_treat = []

    for treatment_pair in treatment_combinations_dict.keys():

        P_values_significant = treatment_combinations_dict[treatment_pair]['P_values_significant']
        genes_significant = treatment_combinations_dict[treatment_pair]['genes_significant']
        delta_n = treatment_combinations_dict[treatment_pair]['delta_n']


        if treatment_pair[0] not in genes_enriched_all_dict:
            genes_enriched_all_dict[treatment_pair[0]] = {}

        if treatment_pair[1] not in genes_enriched_all_dict:
            genes_enriched_all_dict[treatment_pair[1]] = {}

        # positive = first item
        # negative = second item


        if turner_treat in treatment_pair:

            idx = list(treatment_pair).index(turner_treat)

            if idx == 0:
                genes_significant[delta_n>0]
            else:
                genes_significant[delta_n<0]


            genes_treat.extend(genes_significant)



        #print(treatment_pair)
        #print(genes_significant)
        #print(delta_n)

    genes_count = dict(collections.Counter(genes_treat))

    genes_all = [ x for x in genes_count.keys()  if genes_count[x] == 3 ]

    print(turner_treat)
    print(genes_all)
