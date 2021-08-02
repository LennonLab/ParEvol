from __future__ import division
import os, pickle, math, random, itertools

import numpy
import pandas

from asa159 import rcont2
import itertools

import statsmodels.stats.multitest as multitest

import matplotlib.pyplot as plt
import networkx as nx
import scipy.stats as stats


def get_random_matrix(c_in):
    #```GNU Lesser General Public License v3.0 code from https://github.com/maclandrol/FisherExact```
    # f2py -c -m asa159 asa159.f90
    #c = array
    # remove empty columns
    empty_cols = (numpy.where(~c_in.any(axis=0))[0])
    empty_rows = (numpy.where(~c_in.any(axis=1))[0])
    c = numpy.delete(c_in, empty_cols, axis=1)
    c = numpy.delete(c, empty_rows, axis=0)

    key = numpy.array([False], dtype=bool)
    ierror = numpy.array([0], dtype=numpy.int32)
    sr, sc = c.sum(axis=1).astype(numpy.int32), c.sum(axis=0).astype(numpy.int32)
    nr, nc = len(sr), len(sc)
    n = numpy.sum(sr)
    replicate=1000
    results = numpy.zeros(replicate)

    seed=None
    # test to see if we can increase wkslimit for neutral sims!!!!
    #wkslimit=5000
    wkslimit=50000
    DFAULT_MAX_TOT = 5000
    # set default maxtot to wkslimit
    if wkslimit < DFAULT_MAX_TOT:
        wkslimit = 5000
    if seed is None:
        try:
            seed = random.SystemRandom().randint(1, 100000)
            seed = numpy.array([seed], dtype=numpy.int32)
        except:
            try:
                import time
                seed = int(time.time())
                seed = numpy.array([seed], dtype=numpy.int32)
            except:
                seed = 12345
                seed = numpy.array([seed], dtype=numpy.int32)

    if n < wkslimit:
        # we can just set the limit  to the table sum
        wkslimit = n
        pass
    else:
        # throw error immediately
        raise ValueError(
            "Limit of %d on the table sum exceded (%d), please increase workspace !" % (DFAULT_MAX_TOT, n))

    maxtot = numpy.array([wkslimit], dtype=numpy.int32)
    fact = numpy.zeros(wkslimit + 1, dtype=numpy.float, order='F')
    observed = numpy.zeros((nr, nc), dtype=numpy.int32, order='F')

    rcont2(nrow=nr, ncol=nc, nrowt=sr, ncolt=sc, maxtot=maxtot,
           key=key, seed=seed, fact=fact, matrix=observed, ierror=ierror)

    # if we do not have an error, make spcial action
    #ans = 0.
    tmp_observed = observed.ravel()
    if ierror[0] in [1, 2]:
        raise ValueError(
            "Error in rcont2 (fortran) : row or column input size is less than 2!")
    elif ierror[0] in [3, 4]:
        raise ValueError(
            "Error in rcont2 (fortran) : Negative values in table !")
    elif ierror[0] == 6:
        # this shouldn't happen with the previous check
        raise ValueError(
            "Error in rcont2 (fortran) : Limit on the table sum (%d) exceded, please increase workspace !" % DFAULT_MAX_TOT)
    else:

        #for empty_column in empty_c:
        #    numpy.insert(tmp_observed, empty_column, nr, axis=1)
        rndm_matrix = numpy.reshape(tmp_observed, (nr,nc))
        for empty_column in empty_cols:
            rndm_matrix = numpy.insert(rndm_matrix, empty_column, 0, axis=1)
        for empty_row in empty_rows:
            rndm_matrix = numpy.insert(rndm_matrix, empty_row, 0, axis=0)

        return rndm_matrix
        #return numpy.reshape(tmp_observed, (nr,nc))


test_array = numpy.asarray([[1,3,4,0],[8,3,1,8], [4,0,0,3], [3,1,2,0], [3,6,0,0], [3,2,1,0], [3,8,0,0]])



def get_mutual_information_binary_matrix(count_matrix):

    # assume

    count_matrix = numpy.where(count_matrix > 0.5, 1, 0)


    num_samples = count_matrix.shape[1]
    #print(count_matrix.sum(axis=1))

    pseudocount = 1 / sum(count_matrix.sum(axis=0))
    normalization_constant = 1 / (num_samples * (1+pseudocount))
    normalization_constant_pairs =  1 / (num_samples * ((1+pseudocount) ** 2)  )

    pseudocount_count_matrix = count_matrix + pseudocount
    occupancy_probability = normalization_constant * pseudocount_count_matrix.sum(axis=1)

    joint_probability_1_1 = normalization_constant_pairs * numpy.matmul(pseudocount_count_matrix, pseudocount_count_matrix.transpose())
    joint_probability_1_0 = normalization_constant_pairs * numpy.matmul(pseudocount_count_matrix, 1+normalization_constant-pseudocount_count_matrix.transpose())
    joint_probability_0_1 = normalization_constant_pairs * numpy.matmul( 1+normalization_constant-pseudocount_count_matrix, pseudocount_count_matrix.transpose())
    joint_probability_0_0 = normalization_constant_pairs * numpy.matmul( 1+normalization_constant-pseudocount_count_matrix, 1+normalization_constant-pseudocount_count_matrix.transpose())


    independent_probability_1_1 = numpy.outer(occupancy_probability,occupancy_probability.transpose())
    independent_probability_1_0 = numpy.outer(occupancy_probability,(1-occupancy_probability).transpose())
    independent_probability_0_1 = numpy.outer((1-occupancy_probability),occupancy_probability.transpose())
    independent_probability_0_0 = numpy.outer((1-occupancy_probability),(1-occupancy_probability).transpose())


    #print(numpy.divide(joint_probability_1_1, independent_probability_1_1))

    mutual_information_matrix = joint_probability_1_1 * numpy.log2(numpy.divide(joint_probability_1_1, independent_probability_1_1))
    mutual_information_matrix += joint_probability_1_0 * numpy.log2(numpy.divide(joint_probability_1_0, independent_probability_1_0))
    mutual_information_matrix += joint_probability_0_1 * numpy.log2(numpy.divide(joint_probability_0_1, independent_probability_0_1))
    mutual_information_matrix += joint_probability_0_0 * numpy.log2(numpy.divide(joint_probability_0_0, independent_probability_0_0))

    mutual_information_total = (mutual_information_matrix.sum() - numpy.trace(mutual_information_matrix))/2

    return mutual_information_matrix




def get_interactions():

    df_path = '/Users/wrshoemaker/Desktop/ParEvol_test/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pandas.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    df_np = numpy.transpose(df_np)
    genes = df.columns.to_list()

    gene_pairs = list(itertools.combinations(genes,2))

    pairwise_null_dict = {}

    for gene_pair in gene_pairs:
        pairwise_null_dict[gene_pair] = []


    mutal_info_matrix = get_mutual_information_binary_matrix(df_np)
    #mutal_info_matrix = numpy.cov(df_np)
    mutal_info_matrix_flat = mutal_info_matrix[numpy.triu_indices(mutal_info_matrix.shape[0], k = 1)]

    n_simulations = 10000

    for i in range(n_simulations):

        if ( i % 1000 == 0) and (i>0):

            print("%d simulations complete!" % i)

        df_np_null = get_random_matrix(df_np)
        null_mutal_info_matrix = get_mutual_information_binary_matrix(df_np_null)
        #null_mutal_info_matrix = numpy.cov(df_np_null)
        null_mutal_info_matrix_flat = null_mutal_info_matrix[numpy.triu_indices(null_mutal_info_matrix.shape[0], k = 1)]

        for gene_pair_idx, gene_pair in enumerate(gene_pairs):

            pairwise_null_dict[gene_pair].append(null_mutal_info_matrix_flat[gene_pair_idx])


    #print(pairwise_null_dict)

    p_values = []
    for gene_pair_idx, gene_pair in enumerate(gene_pairs):

        null_array = numpy.asarray(pairwise_null_dict[gene_pair])
        observed_mutual_info = mutal_info_matrix_flat[gene_pair_idx]

        p_value_gene_pair =  (len(null_array[null_array > observed_mutual_info ]) + 1) / (n_simulations+1)

        p_values.append(p_value_gene_pair)

    # 63190 tests

    reject, pvals_corrected, alphacSidak, alphacBonf = multitest.multipletests(p_values, alpha=0.05, method='fdr_bh')
    significanat_interaction_dict = {}
    count = 0
    for gene_pair_idx, gene_pair in enumerate(gene_pairs):

        observed_mutual_info = mutal_info_matrix_flat[gene_pair_idx]
        p_value_corrected = pvals_corrected[gene_pair_idx]

        if p_value_corrected >= 0.01:
            continue

        #if reject[gene_pair_idx] == True:
        #    continue

        count += 1

        if gene_pair[0] not in significanat_interaction_dict:
            significanat_interaction_dict[gene_pair[0]] = {}

        if gene_pair[1] not in significanat_interaction_dict:
            significanat_interaction_dict[gene_pair[1]] = {}

        significanat_interaction_dict[gene_pair[0]][gene_pair[1]] = observed_mutual_info
        significanat_interaction_dict[gene_pair[1]][gene_pair[0]] = observed_mutual_info



    df_significant = pandas.DataFrame.from_dict(significanat_interaction_dict)

    df_significant = df_significant.fillna(0)
    df_out = '/Users/wrshoemaker/Desktop/ParEvol_test/data/Tenaillon_et_al/significant_mutual_information_tenaillon.txt'
    df_significant.to_csv(df_out, sep = '\t', index = True)


def plot_mutual_info_network():

    def show_graph_with_labels(adjacency_matrix):
        rows, cols = numpy.where(adjacency_matrix > 0.08)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        gr.add_edges_from(edges)
        #labels=mylabels, with_labels=True
        nx.draw(gr, node_size=10)
        plt.savefig('/Users/wrshoemaker/Desktop/ParEvol_test/test.png'  ,  bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
        plt.close()


    def make_label_dict(labels):
        l = {}
        for i, label in enumerate(labels):
            l[i] = label
        return l

    df_out = '/Users/wrshoemaker/Desktop/ParEvol_test/data/Tenaillon_et_al/significant_mutual_information_tenaillon.txt'

    mydata = numpy.genfromtxt(df_out, delimiter='\t')
    adjacency = mydata[1:,1:]
    print(adjacency)

    show_graph_with_labels(adjacency)





#mean_absolute = []
#variances_absolute = []

def probability_absence(gene, N, mut_counts_dict, zeros=True):

    if zeros == True:

        mean_relative_muts_denom = sum([ mut_counts_dict[g]['mean_relative_muts'] for g in mut_counts_dict.keys() ])
        mean_relative_muts_num = sum([ mut_counts_dict[g]['mean_relative_muts'] for g in mut_counts_dict.keys() if g != gene ])

    else:

        mean_relative_muts_denom = sum([ mut_counts_dict[g]['mean_relative_muts_no_zeros'] for g in mut_counts_dict.keys() ])
        mean_relative_muts_num = sum([ mut_counts_dict[g]['mean_relative_muts_no_zeros'] for g in mut_counts_dict.keys() if g != gene ])


    return (mean_relative_muts_num/mean_relative_muts_denom)**N





df_path = '/Users/wrshoemaker/Desktop/ParEvol_test/data/Tenaillon_et_al/gene_by_pop.txt'
df = pandas.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
genes = df.columns.to_list()
df_np = df.values
df_np = numpy.transpose(df_np)

mut_counts_dict = {}



means = []
variances = []
for gene_counts_idx, gene_counts_i in enumerate(df_np):

    gene_i = genes[gene_counts_idx]

    relative_gene_counts = gene_counts_i / len(df_np.sum(axis=0))

    means.append(numpy.mean(relative_gene_counts))
    variances.append(numpy.var(relative_gene_counts))

    mut_counts_dict[gene_i] = {}
    mut_counts_dict[gene_i]['mean_relative_muts'] = numpy.mean(relative_gene_counts)
    mut_counts_dict[gene_i]['mean_relative_muts_no_zeros'] = numpy.mean(relative_gene_counts[relative_gene_counts>0])

means = numpy.asarray(means)
variances = numpy.asarray(variances)




def plot_obs_pred_occupancy():


    N_array = df_np.sum(axis=0)

    df_np_pres_abs = numpy.where(df_np > 0, 1, 0)
    observed_occupancies = df_np_pres_abs.sum(axis=1) / df_np.shape[1]


    predicted_occupancies = []

    for gene_idx, gene in enumerate(genes):

        #mut_counts_dict[gene]['mean_relative_muts']
        absence_prob_list = [probability_absence(gene, N, zeros=True) for N in N_array]
        predicted_occupancies.append(1-numpy.mean(absence_prob_list))

    predicted_occupancies = numpy.asarray(predicted_occupancies)


    population_idx =  numpy.arange(0, df_np_pres_abs.shape[1], 1)
    n_subsamples = numpy.arange(10, df_np_pres_abs.shape[1], 5)

    subsamples=1

    mae_dict = {}

    for n_i in n_subsamples:

        mae_dict[n_i] = {}

        mean_absolute_error_all = []

        for subsample in range(1000):

            population_idx_subsample = numpy.random.choice(population_idx, size=n_i, replace=False)

            df_np_subsample = df_np[:, population_idx_subsample]

            df_np_pres_abs_subsample = numpy.where(df_np_subsample > 0, 1, 0)

            observed_occupancies_subsample = df_np_pres_abs_subsample.sum(axis=1) / df_np_subsample.shape[1]

            N_subsample_array = df_np_subsample.sum(axis=0)

            predicted_occupancies_subsample = []

            for gene_idx, gene in enumerate(genes):

                absence_prob_subsample_list = [probability_absence(gene, N_subsample, zeros=True) for N_subsample in N_subsample_array if N_subsample> 0 ]
                predicted_occupancies_subsample.append(1-numpy.mean(absence_prob_subsample_list))

            predicted_occupancies_subsample = numpy.asarray(predicted_occupancies_subsample)

            predicted_occupancies_subsample = predicted_occupancies_subsample[ observed_occupancies_subsample>0 ]
            observed_occupancies_subsample = observed_occupancies_subsample[ observed_occupancies_subsample>0 ]


            mean_absolute_error_i = numpy.mean(numpy.absolute(observed_occupancies_subsample-predicted_occupancies_subsample))
            mean_absolute_error_all.append(mean_absolute_error_i)

            #print(n_i, mean_absolute_error_i)

        mean_absolute_error_all = numpy.asarray(mean_absolute_error_all)

        mae_dict[n_i]['mae_mean'] = numpy.mean(mean_absolute_error_all)
        mae_dict[n_i]['mae_025'] = numpy.percentile(mean_absolute_error_all, 0.025)
        mae_dict[n_i]['mae_975'] = numpy.percentile(mean_absolute_error_all, 0.975)

        print(n_i, numpy.percentile(mean_absolute_error_all, 0.025), numpy.percentile(mean_absolute_error_all, 0.975))


    sys.stdout.write("Dumping pickle......\n")
    with open('/Users/wrshoemaker/Desktop/ParEvol_test/subsample_poisson_occupancy.pickle', 'wb') as handle:
        pickle.dump(mae_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.write("Done!\n")






    #fig, ax = plt.subplots(figsize=(4,4))
    #fig = plt.figure(figsize = (4, 4)) #
    #ax_occupancy = plt.subplot2grid((1, 2), (0, 0))
    #ax_occupancy_subsample = plt.subplot2grid((1, 2), (0, 1))

    #ax_taylor = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    #ax_taylor_subsample = plt.subplot2grid((2, 2), (1, 1), colspan=1)




    #ax.plot([0.01,1],[0.01,1], lw=3,ls='--',c='k',zorder=1)
    #ax.scatter(observed_occupancies, predicted_occupancies, c='dodgerblue', alpha=0.8,zorder=2)#, c='#87CEEB')



    #ax.set_xlim([0.007, 1.1])
    #ax.set_ylim([0.007, 1.1])

    #ax.scatter(0.00000001, 0.000000001, alpha=0.8, c='dodgerblue', label='Gene')#, c='#87CEEB')

    #ax.set_title('Proportion of populations\ncontaining  ' + r'$\geq 1$' + ' mutation', fontsize=11, fontweight='bold' )

    #ax.set_xscale('log', base=10)
    #ax.set_yscale('log', base=10)
    #ax.set_xlabel('Observed', fontsize=12)
    #ax.set_ylabel('Predicted', fontsize=12)

    #ax.legend(loc="upper left", fontsize=8)

    #fig.subplots_adjust(wspace=0.3, hspace=0.3)
    #fig.savefig('/Users/wrshoemaker/Desktop/ParEvol_test/pred_obs_occupancies.png', format='png', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    #plt.close()



    #fig, ax = plt.subplots(figsize=(4,4))
    #fig.subplots_adjust(bottom= 0.15)

    # Bhatia–Davis inequality
    #mean_range = numpy.linspace(min(means), max(means), num=1000)
    #variance_range = (1-mean_range) * mean_range
    #ax.plot(mean_range, variance_range, lw=2, ls=':', c = 'k', label='Bhatia–Davis inequality')

    #ax.scatter(means, variances, c='dodgerblue', alpha=0.8,zorder=2)#, c='#87CEEB')

    #slope, intercept, r_value, p_value, std_err = stats.linregress(numpy.log10(means), numpy.log10(variances))


    #ax.text(0.2,0.8, r'$\sigma^{{2}}_{{ x }} \propto \left \langle x \right \rangle^{{{}}}$'.format(str(round(slope, 2)) ), fontsize=12, color='k', ha='center', va='center', transform=ax.transAxes  )


    #x_log10_range =  numpy.linspace(min(numpy.log10(means)) , max(numpy.log10(means)) , 10000)
    #y_log10_fit_range = 10 ** (slope*x_log10_range + intercept)
    #y_log10_null_range = 10 ** (1*x_log10_range + intercept)

    #ax.plot(10**x_log10_range, y_log10_fit_range, c='k', lw=2.5, linestyle='--', zorder=2, label="OLS regression")
    #ax.plot(10**x_log10_range, y_log10_null_range, c='grey', lw=2.5, linestyle='--', zorder=2, label="Taylor's law")

    #ax.legend(loc="upper left", fontsize=8)

    #ax.set_xscale('log', base=10)
    #ax.set_yscale('log', base=10)
    #ax.set_xlabel('Mean proportion of mutations, ' + r'$\left \langle x \right \rangle$', fontsize=12)
    #ax.set_ylabel('Variance of proportion of mutations, ' + r'$\sigma^{2}_{x}$', fontsize=12)

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig('/Users/wrshoemaker/Desktop/ParEvol_test/occupancy_and_slope.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.5, dpi = 600)
    plt.close()





plot_obs_pred_occupancy()
