
from __future__ import division
import os, sys, signal, pickle, operator
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from scipy.stats import itemfreq, gamma, lognorm

from macroeco_distributions import pln, pln_solver, pln_ll
from macroecotools import obs_pred_rsquare



df_path =  '/Users/WRShoemaker/GitHub/ParEvol/data/Tenaillon_et_al/gene_by_pop.txt'
df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)

def plot_lognorm():

    # create multiplicity dicts
    with open('/Users/WRShoemaker/GitHub/ParEvol/data/Tenaillon_et_al/gene_size_dict.txt', 'rb') as handle:
        length_dict = pickle.loads(handle.read())

    # get parallelism statistics
    gene_parallelism_statistics = {}
    for gene_i, length_i in length_dict.items():
        gene_parallelism_statistics[gene_i] = {}
        gene_parallelism_statistics[gene_i]['length'] = length_i
        gene_parallelism_statistics[gene_i]['observed'] = 0
        gene_parallelism_statistics[gene_i]['multiplicity'] = 0


    gene_mut_counts = df.sum(axis=0)
    # save number of mutations for multiplicity
    for locus_tag_i, n_muts_i in gene_mut_counts.iteritems():
        gene_parallelism_statistics[locus_tag_i]['observed'] = n_muts_i

    L_mean = np.mean(list(length_dict.values()))
    L_tot = sum(list(length_dict.values()))
    n_tot = sum(gene_mut_counts.values)
    # go back over and calculate multiplicity
    multiplicities = []
    for locus_tag_i in gene_parallelism_statistics.keys():
        # double check the measurements from this
        gene_parallelism_statistics[locus_tag_i]['multiplicity'] = gene_parallelism_statistics[locus_tag_i]['observed'] *1.0/ length_dict[locus_tag_i] * L_mean
        gene_parallelism_statistics[locus_tag_i]['expected'] = n_tot*gene_parallelism_statistics[locus_tag_i]['length']/L_tot

        if gene_parallelism_statistics[locus_tag_i]['multiplicity'] > 0:
             multiplicities.append(gene_parallelism_statistics[locus_tag_i]['multiplicity'])


    log_multiplicities = np.log(multiplicities)

    rescaled_log_multiplicities = (log_multiplicities - np.mean(log_multiplicities)) / np.std(log_multiplicities)

    ag,bg,cg = lognorm.fit(rescaled_log_multiplicities)

    fig, ax = plt.subplots(figsize=(4,4))

    ax.hist(rescaled_log_multiplicities, alpha=0.8, bins= 20, density=True)#, weights=np.zeros_like(multiplicities) + 1. / len(multiplicities), alpha=0.8, color = '#175ac6')
    x_range = np.linspace(min(rescaled_log_multiplicities) , max(rescaled_log_multiplicities) , 10000)

    ax.plot(x_range, lognorm.pdf(x_range, ag, bg,cg), 'k', lw=2, label = 'Log-normal fit')
    ax.legend(loc="upper right", fontsize=8)

    ax.set_yscale('log', basey=10)

    ax.set_xlabel('Rescaled log multiplicity', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)

    fig.tight_layout()
    fig.savefig('/Users/WRShoemaker/GitHub/ParEvol/figs/mult_hist.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



#for index, row in df.iterrows():
#    mutation_dist = row.values
#    mutation_dist = mutation_dist[mutation_dist>0]
#    print(mutation_dist)


def plot_lognorm():

    AFD = df.sum(axis=1).values

    log_AFD = np.log(AFD)

    rescaled_log_AFD = (log_AFD - np.mean(log_AFD)) / np.std(log_AFD)

    fig, ax = plt.subplots(figsize=(4,4))

    ax.hist(rescaled_log_AFD, alpha=0.8, bins= 20, density=True)#, weights=np.zeros_like(multiplicities) + 1. / len(multiplicities), alpha=0.8, color = '#175ac6')

    ax.set_xlabel('Rescaled log mutations', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)

    fig.tight_layout()
    fig.savefig('/Users/WRShoemaker/GitHub/ParEvol/figs/mutations.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


plot_lognorm()
