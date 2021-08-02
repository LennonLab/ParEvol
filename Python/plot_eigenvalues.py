from __future__ import division
import os, pickle, operator, sys
import random, copy
from itertools import compress
import numpy as np
import pandas as pd
import parevol_tools as pt
import clean_data as cd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


np.random.seed(123456789)
random.seed(123456789)


#pt.mydir

iter = 10000

# code for occupancy Poisson



position_gene_map, effective_gene_lengths_nonsynonymous, effective_gene_lengths_synonymous, substitution_specific_synonymous_fraction = pt.create_annotation_map('tenaillon')

gene_data = pt.parse_gene_list('tenaillon')

gene_names, gene_start_positions, gene_end_positions, promoter_start_positions, promoter_end_positions, gene_sequences, strands, genes, features, protein_ids = gene_data



def map_tenaillon_genes_to_locus_tags(genes_list, gene_names=gene_names, genes=genes):

    locus_tags_non = []

    for gene_i in genes_list:

        if gene_i in gene_names:
            gene_idx = gene_names.index(gene_i)

        if gene_i in genes:
            gene_idx = genes.index(gene_i)

        locus_tags_non.append(gene_names[gene_idx])

    return locus_tags_non



def get_mut_counts_dict(pop_by_gene_np, locus_tags):

    mut_counts_syn_dict = {}

    #means = []
    #variances = []
    for gene_counts_idx, gene_counts_i in enumerate(pop_by_gene_np):

        gene_i = locus_tags[gene_counts_idx]

        relative_gene_counts = gene_counts_i / len(pop_by_gene_np.sum(axis=0))

        #means.append(np.mean(relative_gene_counts))
        #variances.append(np.var(relative_gene_counts))

        mut_counts_syn_dict[gene_i] = {}
        mut_counts_syn_dict[gene_i]['mean_relative_muts'] = np.mean(relative_gene_counts)
        mut_counts_syn_dict[gene_i]['mean_relative_muts_no_zeros'] = np.mean(relative_gene_counts[relative_gene_counts>0])

    #means = np.asarray(means)
    #variances = np.asarray(variances)

    return mut_counts_syn_dict





def probability_absence(gene, N, mut_counts_dict, zeros=True):

    if zeros == True:
        mean_relative_muts_denom = sum([ mut_counts_dict[g]['mean_relative_muts'] for g in mut_counts_dict.keys() ])
        mean_relative_muts_num = sum([ mut_counts_dict[g]['mean_relative_muts'] for g in mut_counts_dict.keys() if g != gene ])

    else:
        mean_relative_muts_denom = sum([ mut_counts_dict[g]['mean_relative_muts_no_zeros'] for g in mut_counts_dict.keys() ])
        mean_relative_muts_num = sum([ mut_counts_dict[g]['mean_relative_muts_no_zeros'] for g in mut_counts_dict.keys() if g != gene ])

    return (mean_relative_muts_num/mean_relative_muts_denom)**N






df_non_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop_nonsyn.txt'
df_non = pd.read_csv(df_non_path, sep = '\t', header = 'infer', index_col = 0)
genes_non = df_non.columns.to_list()
df_non_np = df_non.values
df_non_np = np.transpose(df_non_np)

locus_tags_non = map_tenaillon_genes_to_locus_tags(genes_non)

mut_counts_non_dict = get_mut_counts_dict(df_non_np, locus_tags_non)

#df_syn_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop_syn.txt'
#df_syn = pd.read_csv(df_syn_path, sep = '\t', header = 'infer', index_col = 0)
#genes_syn = df_syn.columns.to_list()
#df_syn_np = df_syn.values
#df_syn_np = np.transpose(df_syn_np)



#locus_tags_syn = map_tenaillon_genes_to_locus_tags(genes_syn)
#mut_counts_syn_dict = get_mut_counts_dict(df_syn_np, locus_tags_syn)



def get_predicted_observed_occupancies(df_np_, df_genes_, mut_counts_dict_, zeros=True):

    N_array_ = df_np_.sum(axis=0)

    df_np_pres_abs_ = np.where(df_np_ > 0, 1, 0)
    observed_occupancies_ = df_np_pres_abs_.sum(axis=1) / df_np_.shape[1]

    predicted_occupancies_ = []

    for gene_idx, gene in enumerate(df_genes_):
        #mut_counts_dict[gene]['mean_relative_muts']
        absence_prob_list_ = [probability_absence(gene, N, mut_counts_dict_, zeros=zeros) for N in N_array_]
        predicted_occupancies_.append(1-np.mean(absence_prob_list_))

    predicted_occupancies_ = np.asarray(predicted_occupancies_)

    predicted_occupancies_ = predicted_occupancies_[ observed_occupancies_>0 ]
    observed_occupancies_ = observed_occupancies_[ observed_occupancies_>0 ]

    return observed_occupancies_, predicted_occupancies_




def calculate_subsampled_mae(df_np, df_genes, mut_counts_dict, subsamples=1, name='non'):

    population_idx =  np.arange(0, df_np.shape[1], 1)
    n_subsamples = np.arange(10, df_np.shape[1], 5)

    print(population_idx)
    print(n_subsamples)

    mae_dict = {}

    for n_i in n_subsamples:

        sys.stdout.write("%d populations......\n" % n_i)

        mae_dict[n_i] = {}

        mean_absolute_error_all = []

        for subsample in range(subsamples):

            population_idx_subsample = np.random.choice(population_idx, size=n_i, replace=False)

            df_np_subsample = df_np[:, population_idx_subsample]

            observed_occupancies_subsample, predicted_occupancies_subsample = get_predicted_observed_occupancies(df_np_subsample, df_genes, mut_counts_dict)

            #df_np_pres_abs_subsample = np.where(df_np_subsample > 0, 1, 0)
            #observed_occupancies_subsample = df_np_pres_abs_subsample.sum(axis=1) / df_np_subsample.shape[1]
            #N_subsample_array = df_np_subsample.sum(axis=0)
            #predicted_occupancies_subsample = []
            #for gene_idx, gene in enumerate(df_genes):

            #    absence_prob_subsample_list = [probability_absence(gene, N_subsample, mut_counts_dict) for N_subsample in N_subsample_array if N_subsample> 0 ]
            #    predicted_occupancies_subsample.append(1-np.mean(absence_prob_subsample_list))

            #predicted_occupancies_subsample = np.asarray(predicted_occupancies_subsample)
            predicted_occupancies_subsample = predicted_occupancies_subsample[ observed_occupancies_subsample>0 ]
            observed_occupancies_subsample = observed_occupancies_subsample[ observed_occupancies_subsample>0 ]

            mean_absolute_error_i = np.mean(np.absolute(observed_occupancies_subsample-predicted_occupancies_subsample))
            mean_absolute_error_all.append(mean_absolute_error_i)


        mean_absolute_error_all = np.asarray(mean_absolute_error_all)

        mae_dict[n_i]['mae_mean'] = np.mean(mean_absolute_error_all)
        mae_dict[n_i]['mae_025'] = np.percentile(mean_absolute_error_all, 2.5)
        mae_dict[n_i]['mae_975'] = np.percentile(mean_absolute_error_all, 97.5)


    sys.stdout.write("Dumping pickle......\n")
    file_name =  '%s/data/Tenaillon_et_al/subsample_poisson_occupancy_%s.pickle' % (pt.get_path(), name)
    with open(file_name, 'wb') as handle:
        pickle.dump(mae_dict, handle)
    sys.stdout.write("Done!\n")






#calculate_subsampled_mae(df_non_np, locus_tags_non, mut_counts_non_dict, subsamples=1000, name='non')
#calculate_subsampled_mae(df_syn_np, locus_tags_syn, mut_counts_syn_dict, subsamples=1000, name='syn')


with open('%s/data/Tenaillon_et_al/subsample_poisson_occupancy_%s.pickle' % (pt.get_path(), 'non'), 'rb') as handle:
    mae_dict_non = pickle.load(handle)

#with open('%s/data/Tenaillon_et_al/subsample_poisson_occupancy_%s.pickle' % (pt.get_path(), 'syn'), 'rb') as handle:
#    mae_dict_syn = pickle.load(handle)


mae_non_x = list(mae_dict_non.keys())
mae_non_x.sort()
mae_mean_non_y = [mae_dict_non[x]['mae_mean'] for x in mae_non_x]
mae_mean_non_y_975 = [mae_dict_non[x]['mae_975'] for x in mae_non_x]
mae_mean_non_y_025 = [mae_dict_non[x]['mae_025'] for x in mae_non_x]

mae_non_x = np.asarray(mae_non_x)
mae_mean_non_y = np.asarray(mae_mean_non_y)
mae_mean_non_y_975 = np.asarray(mae_mean_non_y_975)
mae_mean_non_y_025 = np.asarray(mae_mean_non_y_025)

#mae_syn_x = list(mae_dict_syn.keys())
#mae_syn_x.sort()
#mae_mean_syn_y = [mae_dict_syn[x]['mae_mean'] for x in mae_syn_x]
#mae_mean_syn_y_975 = [mae_dict_syn[x]['mae_975'] for x in mae_syn_x]
#mae_mean_syn_y_025 = [mae_dict_syn[x]['mae_025'] for x in mae_syn_x]

#mae_syn_x = np.asarray(mae_syn_x)
#mae_mean_syn_y = np.asarray(mae_mean_syn_y)
#mae_mean_syn_y_975 = np.asarray(mae_mean_syn_y_975)
#mae_mean_syn_y_025 = np.asarray(mae_mean_syn_y_025)



observed_occupancies_non, predicted_occupancies_non = get_predicted_observed_occupancies(df_non_np, locus_tags_non, mut_counts_non_dict)
#observed_occupancies_syn, predicted_occupancies_syn = get_predicted_observed_occupancies(df_syn_np, locus_tags_syn, mut_counts_syn_dict)





df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop_nonsyn.txt'
df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
df_np = df.values
gene_names = df.columns.values
n_rows = list(range(df_np.shape[0]))

df_np_delta = cd.likelihood_matrix_array(df_np, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()

X = df_np_delta/df_np_delta.sum(axis=1)[:,None]
X = X - np.mean(X, axis = 0)
#cov = np.cov(X.T)
#ev, eig = np.linalg.eig(cov)
pca = PCA()
pca_fit = pca.fit_transform(X)
#L = pt.get_L_stat(max(ev), N, cov.shape[0])
eig = pt.get_x_stat(pca.explained_variance_[:-1], n_features=X.shape[1])

eig_null = []
for j in range(iter):
    df_np_j = pt.get_random_matrix(df_np)
    np.seterr(divide='ignore')
    df_np_j_delta = cd.likelihood_matrix_array(df_np_j, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()
    X_j = df_np_j_delta/df_np_j_delta.sum(axis=1)[:,None]
    X_j -= np.mean(X_j, axis = 0)
    pca_j = PCA()
    pca_X_j = pca_j.fit_transform(X_j)
    eig_null.append( pt.get_x_stat(pca_j.explained_variance_[:-1], n_features=X.shape[1]) )


eig_null = np.asarray(eig_null)

P_eig = len(eig_null[eig_null > eig]) / len(eig_null)


eig_power = open(pt.get_path() + '/data/Tenaillon_et_al/power_sample_size_l_stat.txt', 'r')

eig_power_pops = []
eig_power_mean = []
eig_power_025 = []
eig_power_975 = []
eig_power.readline()
for line in eig_power:
    line = line.strip().split('\t')
    eig_power_pops.append(float(line[0]))
    eig_power_mean.append(float(line[1]))
    eig_power_025.append(float(line[2]))
    eig_power_975.append(float(line[3]))

eig_power.close()

eig_power_pops = np.asarray(eig_power_pops)
eig_power_mean = np.asarray(eig_power_mean)
eig_power_025 = np.asarray(eig_power_025)
eig_power_975 = np.asarray(eig_power_975)




fig = plt.figure(figsize = (8, 8)) #

ax_poisson_occupancy = plt.subplot2grid((2, 2), (0, 0))
ax_poisson_mae = plt.subplot2grid((2, 2), (0, 1))
ax_eigen_hist = plt.subplot2grid((2, 2), (1, 0))
ax_eigen_subsample = plt.subplot2grid((2, 2), (1, 1))


ax_poisson_occupancy.text(-0.1, 1.07,'a', fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_poisson_occupancy.transAxes)
ax_poisson_mae.text(-0.1, 1.07,'b', fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_poisson_mae.transAxes)
ax_eigen_hist.text(-0.1, 1.07,'c', fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_eigen_hist.transAxes)
ax_eigen_subsample.text(-0.1, 1.07,'d', fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_eigen_subsample.transAxes)



ax_eigen_hist.hist(eig_null, bins=20, color='dodgerblue', density=True)

ax_eigen_hist.axvline(x=eig, ls='--', c='k', lw=2)
ax_eigen_hist.set_xlabel('Primary eigenvalue, ' + r'$\tilde{L}_{1}$', fontsize=12)
ax_eigen_hist.set_ylabel('Probability density', fontsize=12)

ax_eigen_hist.text(0.15, 0.9, r'$P = $' + str(round(P_eig, 3)), fontsize=11, ha='center', va='center', transform=ax_eigen_hist.transAxes)



ax_eigen_subsample.scatter(eig_power_pops, eig_power_mean, c='dodgerblue')
ax_eigen_subsample.axhline(y=P_eig, ls='--', lw=2, color='k', label='All populations')
ax_eigen_subsample.axhline(y=0.05, ls=':', lw=2, color='grey', label=r'$\alpha = 0.05$')

#ax_eigen_subsample.axhline(y=0.5, ls=':', lw=2, color='grey', label="No covariance")

ax_eigen_subsample.set_xlabel('Number of replicate populations', fontsize=12)
ax_eigen_subsample.set_ylabel('Prop. null primary eigenvalues\n$>$ primary eigenvalue', fontsize=9)
ax_eigen_subsample.set_ylim([-0.005, 0.65])

ax_eigen_subsample.errorbar(eig_power_pops, eig_power_mean, yerr = [eig_power_mean - eig_power_975, eig_power_025 - eig_power_mean ], \
    fmt = 'o', alpha = 1, \
    barsabove = True, marker = '.',  ls = "None", mfc = 'k', mec = 'k', c = 'k', zorder=2)

ax_eigen_subsample.legend(loc="upper right", fontsize=8)





ax_poisson_occupancy.plot([0.01,1],[0.01,1], lw=3,ls='--',c='k',zorder=1)
ax_poisson_occupancy.scatter(observed_occupancies_non, predicted_occupancies_non, c='dodgerblue', alpha=0.8,zorder=2)#, c='#87CEEB')
ax_poisson_occupancy.set_xlim([0.007, 1.1])
ax_poisson_occupancy.set_ylim([0.007, 1.1])
ax_poisson_occupancy.scatter(0.00000001, 0.000000001, alpha=0.8, c='dodgerblue', label='Gene')#, c='#87CEEB')

#ax_poisson_occupancy.set_title('Nonsynonymous mutations', fontsize=11, fontweight='bold' )
ax_poisson_occupancy.set_xscale('log', base=10)
ax_poisson_occupancy.set_yscale('log', base=10)
ax_poisson_occupancy.set_xlabel('Observed occupancy', fontsize=12)
ax_poisson_occupancy.set_ylabel('Predicted occupancy, Poisson', fontsize=12)

ax_poisson_occupancy.legend(loc="upper left", fontsize=8)

mad_all = np.mean(np.absolute(observed_occupancies_non - predicted_occupancies_non))

ax_poisson_occupancy.text(0.18, 0.82, r'$\mathrm{MAE}=$' + str(round(mad_all, 3)), fontsize=10, ha='center', va='center', transform=ax_poisson_occupancy .transAxes)




ax_poisson_mae.scatter(mae_non_x, mae_mean_non_y, c='dodgerblue')
ax_poisson_mae.axhline(y=mad_all, ls='--', lw=2, color='k', label= r'$\mathrm{MAE}$' + ' of all populations')
ax_poisson_mae.axhline(y=0, ls=':', lw=2, color='grey')
ax_poisson_mae.set_xlabel('Number of replicate populations', fontsize=12)
ax_poisson_mae.set_ylabel('Mean absolute error of occupancy', fontsize=12)
ax_poisson_mae.set_ylim([-0.005, 0.13])


ax_poisson_mae.errorbar(mae_non_x, mae_mean_non_y, yerr = [mae_mean_non_y - mae_mean_non_y_975, mae_mean_non_y_025 - mae_mean_non_y ], \
    fmt = 'o', alpha = 1, \
    barsabove = True, marker = '.',  ls = "None", mfc = 'k', mec = 'k', c = 'k', zorder=2)


ax_poisson_mae.legend(loc="upper right", fontsize=8)

fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig('%s/figs/eigen_occupancy.pdf' % pt.get_path(), format='pdf', bbox_inches = "tight", pad_inches = 0.2, dpi = 600)
plt.close()
