from __future__ import division
import os, pickle
from itertools import groupby
from operator import itemgetter
import numpy as np
import pandas as pd

mydir = os.path.expanduser("~/GitHub/ParEvol/")


class good_et_al:

    def __init__(self):
        self.populations = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', \
                        'p1', 'p2', 'p3', 'p4', 'p5', 'p6']

    def parse_convergence_matrix(self, filename):

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


    def reformat_convergence_matrix(self):
        conv_dict = self.parse_convergence_matrix(mydir + "data/Good_et_al/gene_convergence_matrix.txt")
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
        df_out = mydir + 'data/Good_et_al/gene_by_pop.txt'
        df.to_csv(df_out, sep = '\t', index = True)


    #def get_enrichment_factors(self):
    #    df_in = mydir + 'data/ltee/gene_by_pop.txt'
    #    df = pd.read_csv(df_in, sep = '\t', header = 'infer', index_col = 0)
    #    # get genes that are significanty enriched for all populations
    #    enriched_df = pd.read_csv(mydir + 'data/ltee/nature24287-s5.csv', sep = ',', header = 'infer')
    #    genes = df.columns.tolist()
    #    enriched_genes_names = enriched_df.Gene.tolist()
    #    genes_I = list(set(genes) & set(enriched_genes_names))
    #
    #    genes_lengths = self.get_gene_lengths(gene_list = genes)
    #    gene_I_lengths = self.get_gene_lengths(gene_list = genes_I)
    #
    #    L_mean = np.mean(list(genes_lengths.values()))
    #    L_i = np.asarray(list(genes_lengths.values()))
    #    N_genes = len(genes)
    #    m_mean = df.sum(axis=1) / N_genes
    #
    #    df_I_n_i = df[genes_I].sum(axis=1)
    #    df_n_i = df.sum(axis=1)
    #
    #    for index, row in df.iterrows():
    #        m_mean_j = m_mean[index]
    #        r_j = (row * (L_mean / L_i)) / m_mean_j
    #        df.loc[index,:] = r_j
    #
    #    df_I = df[genes_I]
    #    L_I_sum = sum(list(gene_I_lengths.values()))
    #    for index, row in df_I.iterrows():
    #        r_j_I = row * ((1 - (L_I_sum / (L_mean * N_genes))) / (1 - ( df_I_n_i[index] / df_n_i[index])))
    #        df_I.loc[index,:] = r_j_I
    #    df_I = df_I.fillna(0)
    #    df_I = df_I.replace([np.inf, -np.inf], 0)
    #    #df_I = df_I.fillna(1)
    #    #df_I = df_I.replace([np.inf, -np.inf], 1)
    #    #df_I.apply(lambda s: s[np.isfinite(s)].dropna())
    #    df_I.loc[:, (df_I != 0).any(axis=0)]
    #
    #    df_I.to_csv(mydir + 'data/ltee/gene_by_pop_m_I.txt', sep = '\t', index = True)

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


    #def get_likelihood_matrix(self):
    #    df_in = mydir + 'data/ltee/gene_by_pop.txt'
    #    df = pd.read_csv(df_in, sep = '\t', header = 'infer', index_col = 0)
    #    genes = df.columns.tolist()
    #    genes_lengths = self.get_gene_lengths(gene_list = genes)
    #    L_mean = np.mean(list(genes_lengths.values()))
    #    L_i = np.asarray(list(genes_lengths.values()))
    #    N_genes = len(genes)
    #    m_mean = df.sum(axis=1) / N_genes


    #    for index, row in df.iterrows():
    #        m_mean_j = m_mean[index]
    #        delta_j = row * np.log((row * (L_mean / L_i)) / m_mean_j)
    #        df.loc[index,:] = delta_j

    #    out_name = 'data/ltee/gene_by_pop_delta.txt'

    #    df_new = df.fillna(0)
    #    # remove colums with all zeros
    #    df_new.loc[:, (df_new != 0).any(axis=0)]
    #    # replace negative values with zero
    #    df_new[df_new < 0] = 0
    #    df_new.to_csv(mydir + out_name, sep = '\t', index = True)


class tenaillon_et_al:

    def clean_tenaillon_et_al(self):
        df_in = mydir + 'data/Tenaillon_et_al/1212986tableS2.csv'
        df_out = open(mydir + 'data/Tenaillon_et_al/1212986tableS2_clean.csv', 'w')
        category_dict = {}
        header = ['Lines', 'Position', 'Type', 'Change', 'Genic_status', 'Gene_nb', \
                    'Gene_name', 'Effect', 'Site_affected', 'Length', \
                    'Genic_type', 'Gene_nb_type', 'Gene_name_type', \
                    'Effect_type', 'Site_affected_type', 'Length_type']
        df_out.write(','.join(header) + '\n')
        # For genic, check whether genic + '_' + 7th column value in dict, if not, select
        # 'Genic' as key
        head_type = { 'Genic': ['Genic', 'Gene_nb', 'Gene_Name', 'Effect', 'codon_affected', 'gene_length_in_codon'] , \
                'Genic_Large_Deletion': ['Genic', 'Gene_nb', 'Gene_Name', 'Large_Deletion', 'bp_deleted_in_Gene', 'gene_length_bp' ], \
                'Genic_RNA':  ['Genic', 'Gene_nb', 'Gene_Name', 'RNA', 'bp_affected', 'gene_length_bp']  ,\
                'Intergenic_Intergenic': ['Intergenic', 'Previous_Gene_nb', 'Previous_Gene_Name_distance_bp', 'Effect', 'Next_Gene_Name_distance_bp', 'Intergenic_type'], \
                'Multigenic_Multigenic': ['Multigenic', 'First_Gene_nb', 'First_Gene_Name', 'Effect', 'Last_Gene_nb', 'Last_Gene_Name']}
        for i, line in enumerate(open(df_in, 'r')):
            line = line.strip().split(',')
            if (len(line) == 0) or (i in range(0, 5)) or (len(line[0]) == 0):
                continue
            else:
                line_type = line[4] + '_' + line[7]
                if line_type in head_type:
                    line_new = line + head_type[line_type]
                else:
                    line_new = line + head_type['Genic']
                df_out.write(','.join(line_new) + '\n')
        df_out.close()

    def pop_by_gene_tenaillon(self):
        pop_by_gene_dict = {}
        gene_size_dict = {}
        df_in = mydir + 'data/Tenaillon_et_al/1212986tableS2_clean.csv'
        for i, line in enumerate(open(df_in, 'r')):
            line_split = line.strip().split(',')
            if (line_split[4] == 'Intergenic') or \
            (i == 0) or \
            (line_split[9].isdigit() == False):
                continue
            gene_length_units = line_split[-1]
            gene_name = line_split[6]
            pop_name = line_split[0]
            if gene_length_units == 'gene_length_in_codon':
                gene_length = int(line_split[9]) * 3
            elif gene_length_units == 'gene_length_bp':
                gene_length = int(line_split[9])
            if gene_name not in gene_size_dict:
                gene_size_dict[gene_name] = gene_length

            if gene_name not in pop_by_gene_dict:
                pop_by_gene_dict[gene_name] = {}

            if pop_name not in pop_by_gene_dict[gene_name]:
                pop_by_gene_dict[gene_name][pop_name] = 1
            else:
                pop_by_gene_dict[gene_name][pop_name] += 1

        df = pd.DataFrame.from_dict(pop_by_gene_dict)
        df = df.fillna(0)
        # remove rows and columns with all zeros
        #df = df.loc[(df.sum(axis=1) != 0), (df.sum(axis=0) != 0)]
        df_out = mydir + 'data/Tenaillon_et_al/gene_by_pop.txt'
        df.to_csv(df_out, sep = '\t', index = True)
        gene_size_dict_out = mydir + 'data/Tenaillon_et_al/gene_size_dict.txt'
        with open(gene_size_dict_out, 'wb') as handle:
            pickle.dump(gene_size_dict, handle)


class kryazhimskiy_et_al:

    def pop_by_gene_kryazhimskiy(self):
        df_path = mydir + 'data/Kryazhimskiy_et_al/NIHMS658386-supplement-Table_S5.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)




#good_et_al().reformat_convergence_matrix()
#good_et_al().get_likelihood_matrix()
#likelihood_matrix('Tenaillon_et_al').get_likelihood_matrix()

#print(kryazhimskiy_et_al().pop_by_gene_kryazhimskiy())
