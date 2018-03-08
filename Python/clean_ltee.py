from __future__ import division
import os
from itertools import groupby
from operator import itemgetter

import numpy as np
import pandas as pd

mydir = os.path.expanduser("~/GitHub/ParEvol/")

populations = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', \
                'p1', 'p2', 'p3', 'p4', 'p5', 'p6']


def make_SbyS(line = 'p1'):
    df_path = mydir + 'data/ltee/ltee_trajectories/' + line + '_annotated_timecourse.txt'
    df = pd.read_csv(df_path, sep = ', ')
    #print(df[['Passed?']])
    df_pass = df.loc[df['Passed?'] == 'PASS']
    print(df_pass.shape)



def parse_well_mixed_state_timecourse(population):

    #haplotype_filename = data_directory+('%s_well_mixed_state_timecourse.txt' % population)
    haplotype_filename = mydir + 'data/ltee/ltee_trajectories/' + population + '_well_mixed_state_timecourse.txt'

    haplotype_file = open(haplotype_filename,"r")

    times = np.array([float(item) for item in haplotype_file.readline().split(",")])
    num_unborn = np.array([float(item) for item in haplotype_file.readline().split(",")])
    num_extinct = np.array([float(item) for item in haplotype_file.readline().split(",")])
    num_fixed = np.array([float(item) for item in haplotype_file.readline().split(",")])
    num_polymorphic = np.array([float(item) for item in haplotype_file.readline().split(",")])

    states = []
    for line in haplotype_file:
        Ls = np.array([float(item) for item in line.split(",")])
        states.append(Ls)
    haplotype_file.close()
    return times, states

def parse_convergence_matrix(filename):

    convergence_matrix = {}

    convergence_matrix_file = open(filename,"r")
    # Header line
    line = convergence_matrix_file.readline()
    populations = [item.strip() for item in line.split(",")[2:]]

    for line in convergence_matrix_file:

        items = line.split(",")
        gene_name = items[0].strip()
        length = float(items[1])

        convergence_matrix[gene_name] = {'length':length, 'mutations': {population: [] for population in populations}}

        for population, item in zip(populations,items[2:]):
            if item.strip()=="":
                continue

            subitems = item.split(";")
            for subitem in subitems:
                subsubitems = subitem.split(":")
                mutation = (float(subsubitems[0]), float(subsubitems[1]), float(subsubitems[2]), float(subsubitems[3]))
                convergence_matrix[gene_name]['mutations'][population].append(mutation)


    return convergence_matrix



def reformat_convergence_matrix():
    conv_dict = parse_convergence_matrix(mydir + "data/ltee/gene_convergence_matrix.txt")
    time_points = []
    new_dict = {}
    for gene_name, gene_data in conv_dict.items():
        for pop_name, mutations in gene_data['mutations'].items():
            for mutation in mutations:
                time = int(mutation[0])
                time_points.append(time)
    time_points = sorted(list(set(time_points)))
    for gene_name, gene_data in conv_dict.items():
        if gene_name not in new_dict:
            new_dict[gene_name] = {}
        for pop_name, mutations in gene_data['mutations'].items():
            if len(mutations) == 0:
                continue
            mutations.sort(key=lambda tup: tup[0])
            # keep only fixed mutations
            #{'A':0,'E':1,'F':2,'P':3}
            mutations = [x for x in mutations if int(x[1]) == 2]
            if len(mutations) == 0:
                continue
            for mutation in mutations:
                time = mutation[0]
                remaining_time_points = time_points[time_points.index(time):]
                #print(time, remaining_time_points)
                for time_point in remaining_time_points:
                    pop_time = pop_name +'_' + str(int(time_point))
                    if pop_time not in new_dict[gene_name]:
                        new_dict[gene_name][pop_time] = 1
                    else:
                        new_dict[gene_name][pop_time] += 1

    df = pd.DataFrame.from_dict(new_dict)
    df = df.fillna(0)
    df = df.loc[:, (df != 0).any(axis=0)]
    df_out = mydir + 'data/ltee/gene_by_pop.txt'
    df.to_csv(df_out, sep = '\t', index = True)

def get_gene_lengths(**keyword_parameters):
    conv_dict = parse_convergence_matrix(mydir + "data/ltee/gene_convergence_matrix.txt")
    length_dict = {}
    if ('gene_list' in keyword_parameters):
        for gene_name in keyword_parameters['gene_list']:
            length_dict[gene_name] = conv_dict[gene_name]['length']
        #for gene_name, gene_data in conv_dict.items():
    else:
        for gene_name, gene_data in conv_dict.items():
            length_dict[gene_name] = conv_dict[gene_name]['length']


    return(length_dict)


def get_enrichment_factors():
    df_in = mydir + 'data/ltee/gene_by_pop.txt'
    df = pd.read_csv(df_in, sep = '\t', header = 'infer', index_col = 0)
    # get genes that are significanty enriched for all populations
    enriched_df = pd.read_csv(mydir + 'data/ltee/nature24287-s5.csv', sep = ',', header = 'infer')
    genes = df.columns.tolist()
    enriched_genes_names = enriched_df.Gene.tolist()
    genes_I = list(set(genes) & set(enriched_genes_names))

    genes_lengths = get_gene_lengths(gene_list = genes)
    gene_I_lengths = get_gene_lengths(gene_list = genes_I)

    L_mean = np.mean(list(genes_lengths.values()))
    L_i = np.asarray(list(genes_lengths.values()))
    N_genes = len(genes)
    m_mean = df.sum(axis=1) / N_genes

    df_I_n_i = df[genes_I].sum(axis=1)
    df_n_i = df.sum(axis=1)

    for index, row in df.iterrows():
        m_mean_j = m_mean[index]
        r_j = (row * (L_mean / L_i)) / m_mean_j
        df.loc[index,:] = r_j

    df_I = df[genes_I]
    L_I_sum = sum(list(gene_I_lengths.values()))
    for index, row in df_I.iterrows():
        r_j_I = row * ((1 - (L_I_sum / (L_mean * N_genes))) / (1 - ( df_I_n_i[index] / df_n_i[index])))
        df_I.loc[index,:] = r_j_I
    df_I = df_I.fillna(0)
    df_I = df_I.replace([np.inf, -np.inf], 0)
    #df_I.apply(lambda s: s[np.isfinite(s)].dropna())
    df_I.loc[:, (df_I != 0).any(axis=0)]

    df_I.to_csv(mydir + 'data/ltee/gene_by_pop_m_I.txt', sep = '\t', index = True)

    #df_new_gene_lengths = get_gene_lengths(gene_list = genes_union)
    #n_genes = len(genes_union)
    #L_mean = np.mean(list(df_new_gene_lengths.values()))
    #L_i = np.asarray(list(df_new_gene_lengths.values()))
    #L_sum = L_i
    #m_mean = df_new.sum(axis=1) / n_genes
    #for index, row in df_new.iterrows():
    #    m_mean_j = m_mean[index]
    #    m_j = row * np.log((row * (L_mean / L_i)) / m_mean_j)
    #    df_new.loc[index,:] = m_j

    #df_new = df_new.fillna(0)
    #df_new.loc[:, (df_new != 0).any(axis=0)]
    #df_new.to_csv(mydir + 'data/ltee/gene_by_pop_m.txt', sep = '\t', index = True)



#reformat_convergence_matrix()
get_enrichment_factors()
