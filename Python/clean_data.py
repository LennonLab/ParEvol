from __future__ import division
import os, pickle, collections, re
from itertools import groupby
from operator import itemgetter
import numpy as np
import pandas as pd
import parevol_tools as pt
from Bio import SeqIO

mydir = os.path.expanduser("~/GitHub/ParEvol/")



class wannier_et_al:


    def get_gene_lengths(self):
        gene_size_dict = {}
        # these are pseudogenes
        to_ignore = ['insN', 'crl', 'yaiX', 'yaiT', 'renD', 'nmpC', 'lomR', \
                    'ydbA', 'ydfJ', 'yoeA', 'wbbL', 'gatR', 'yejO', 'yqiG', \
                    'yhcE', 'yrhA', 'yhiS', 'insO']
        with open(mydir + '/data/Wannier_et_al/NC_000913_3.gb', "rU") as input_handle:
            for record in SeqIO.parse(input_handle, "genbank"):
                for feature in record.features:
                    if feature.type == 'gene':
                        gene_name = feature.qualifiers['gene'][0]
                        if gene_name in to_ignore:
                            continue
                        start_stop = (re.findall(r"[\w']+", str(feature.location)))
                        gene_len = int(start_stop[1]) - int(start_stop[0])
                        gene_size_dict[gene_name] = gene_len

        gene_size_dict_out = mydir + 'data/Wannier_et_al/gene_size_dict.txt'
        with open(gene_size_dict_out, 'wb') as handle:
            pickle.dump(gene_size_dict, handle)

    def clean_data(self):
        # ALEdb setup
        # A = ALE
        # F = Flask
        # I = isolate number
        # R = Technical replicate
        treats = ['C321', 'C321.deltaA', 'C321.deltaA.earlyfix', 'ECNR2.1']
        #treats = ['C321.deltaA.earlyfix']
        table = str.maketrans(dict.fromkeys('""'))
        #gene_site_dict = {}
        for treat in treats:
            gene_treat_dict = {}
            pop_position_dict = {}
            lengths = []
            treat_path = mydir + '/data/Wannier_et_al/' + treat + '_mutation_table.txt'
            for i, line in enumerate(open(treat_path, 'r')):
                line = line.strip('\n')
                items = line.split("\t")
                items = [item.translate(table) for item in items]
                if i == 0:
                    samples = [x.split('>')[1][:-3] for x in items[5:]]
                    samples = [x.replace(' ', '-') for x in samples]
                    if (treat == 'C321.deltaA') or (treat == 'C321.deltaA.earlyfix'):
                        samples_list = [list(x) for x in samples]
                        for x in samples_list:
                            x[5] = 'delta'
                        samples = [''.join(x) for x in samples_list]
                    samples_no_FIR = [s.rsplit('-', 3)[0] for s in samples]
                    for sample_no_FIR in list(set(samples_no_FIR)):
                        gene_treat_dict[sample_no_FIR] = {}
                    for j, sample_no_FIR in enumerate(samples_no_FIR):
                        pop_position_dict[j] = sample_no_FIR

                # remove noncoding and mutations in overlaping genes
                if ('noncoding' in items[4]) or ('pseudogene' in items[4]) \
                        or ('intergenic' in items[4]) or (';' in items[3]) \
                        or (len(set(items[5:])) == 1):
                    continue

                just_muts = items[5:]
                gene = items[3]
                if len(gene.split(',')) > 1:
                    continue
                #gene_site_dict[gene] = items[0]
                # merge reps
                site_dict = {}
                for m, mut_m in enumerate(just_muts):
                    if mut_m == '1':
                        site_dict[pop_position_dict[m]] = 1

                if len(site_dict) > 3:
                    continue
                for key, value in site_dict.items():
                    if gene in gene_treat_dict[key]:
                        gene_treat_dict[key][gene] += 1
                    else:
                        gene_treat_dict[key][gene] = 1

            df = pd.DataFrame.from_dict(gene_treat_dict).T
            df = df.fillna(0)
            df_out = mydir + '/data/Wannier_et_al/' + treat + '_mutation_table_clean.txt'
            df.to_csv(df_out, sep = '\t', index = True)




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


    def reformat_convergence_matrix(self, mut_type = 'F'):
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
                if mut_type == 'F':
                    mutations = [x for x in mutations if int(x[1]) == 2]
                elif mut_type == 'P':
                    mutations = [x for x in mutations if (int(x[1]) == 3) ]#or (int(x[1]) == 0)]
                else:
                    print("Argument mut_type not recognized")

                if len(mutations) == 0:
                    continue
                for mutation in mutations:
                    if mut_type == 'F':
                        time = mutation[0]
                        remaining_time_points = time_points[time_points.index(time):]
                        for time_point in remaining_time_points:
                            pop_time = pop_name +'_' + str(int(time_point))
                            if pop_time not in new_dict[gene_name]:
                                new_dict[gene_name][pop_time] = 1
                            else:
                                new_dict[gene_name][pop_time] += 1
                    elif mut_type == 'P':
                        pop_time = pop_name +'_' + str(int(mutation[0]))
                        if pop_time not in new_dict[gene_name]:
                            new_dict[gene_name][pop_time] = 1
                        else:
                            new_dict[gene_name][pop_time] += 1

        df = pd.DataFrame.from_dict(new_dict)
        df = df.fillna(0)
        df = df.loc[:, (df != 0).any(axis=0)]
        if mut_type == 'F':
            df_out = mydir + 'data/Good_et_al/gene_by_pop.txt'
            #df_delta_out = mydir + 'data/Good_et_al/gene_by_pop_delta.txt'
        elif mut_type == 'P':
            df_out = mydir + 'data/Good_et_al/gene_by_pop_poly.txt'
            #df_delta_out = mydir + 'data/Good_et_al/gene_by_pop_poly_delta.txt'
        else:
            print("Argument mut_type not recognized")
        df.to_csv(df_out, sep = '\t', index = True)






class tenaillon_et_al:

    def clean_data(self):
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
        pop_by_gene_dict_nonsyn = {}
        pop_by_gene_dict_syn = {}
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


            # add code for nonsyn vs syn

            if line_split[4] == 'Genic':
                if ('Synonymous' in line_split[7]) and ('NonSynonymous' not in line_split[7]):
                    if gene_name not in pop_by_gene_dict_syn:
                        pop_by_gene_dict_syn[gene_name] = {}

                    if pop_name not in pop_by_gene_dict_syn[gene_name]:
                        pop_by_gene_dict_syn[gene_name][pop_name] = 1
                    else:
                        pop_by_gene_dict_syn[gene_name][pop_name] += 1

                else:
                    if gene_name not in pop_by_gene_dict_nonsyn:
                        pop_by_gene_dict_nonsyn[gene_name] = {}

                    if pop_name not in pop_by_gene_dict_nonsyn[gene_name]:
                        pop_by_gene_dict_nonsyn[gene_name][pop_name] = 1
                    else:
                        pop_by_gene_dict_nonsyn[gene_name][pop_name] += 1


        df = pd.DataFrame.from_dict(pop_by_gene_dict)
        df = df.fillna(0)
        # original data has "LIne" instead of "Line"
        df.index = df.index.str.replace('LIne', 'Line', regex=True)
        df_out = mydir + 'data/Tenaillon_et_al/gene_by_pop.txt'
        df.to_csv(df_out, sep = '\t', index = True)

        df_nonsyn = pd.DataFrame.from_dict(pop_by_gene_dict_nonsyn)
        df_nonsyn = df_nonsyn.fillna(0)
        df_nonsyn.index = df_nonsyn.index.str.replace('LIne', 'Line', regex=True)
        df_nonsyn_out = mydir + 'data/Tenaillon_et_al/gene_by_pop_nonsyn.txt'
        df_nonsyn.to_csv(df_nonsyn_out, sep = '\t', index = True)

        df_syn = pd.DataFrame.from_dict(pop_by_gene_dict_syn)
        df_syn = df_syn.fillna(0)
        df_syn.index = df_syn.index.str.replace('LIne', 'Line', regex=True)
        df_syn_out = mydir + 'data/Tenaillon_et_al/gene_by_pop_syn.txt'
        df_syn.to_csv(df_syn_out, sep = '\t', index = True)


        gene_size_dict_out = mydir + 'data/Tenaillon_et_al/gene_size_dict.txt'
        with open(gene_size_dict_out, 'wb') as handle:
            pickle.dump(gene_size_dict, handle)



class turner_et_al:

    def clean_data(self):
        genome_path = mydir + 'data/Turner_et_al/GCF_000203955.1_ASM20395v1_genomic.gbff'
        gene_to_locus_tag = {}
        old_locus_tag_to_locus_tag = {}
        locus_tag_size = {}
        # get gene name dictionaries
        for record in SeqIO.parse(genome_path, "genbank"):
            for feature in record.features:
                if 'note' in feature.qualifiers:
                    if 'incomplete' in feature.qualifiers['note'][0]:
                        continue
                    if 'frameshifted' in feature.qualifiers['note'][0]:
                        continue
                    if 'internal stop' in feature.qualifiers['note'][0]:
                        continue
                    if 'riboswitch' in feature.qualifiers['note'][0]:
                        continue
                    locus_tag = feature.qualifiers['locus_tag'][0]
                    nuc_str = str(feature.location.extract(record).seq[:-3])
                    locus_tag_size[locus_tag] = len(nuc_str)

                    if 'gene' in feature.qualifiers:
                        gene_name = feature.qualifiers['gene'][0]
                        gene_to_locus_tag[gene_name] = locus_tag
                    if 'old_locus_tag' in feature.qualifiers:
                        old_locus_tag_name = feature.qualifiers['old_locus_tag'][0]
                        old_locus_tag_to_locus_tag[old_locus_tag_name] = locus_tag

        gene_size_dict_out = mydir + 'data/Turner_et_al/gene_size_dict.txt'
        with open(gene_size_dict_out, 'wb') as handle:
            pickle.dump(locus_tag_size, handle)

        df_in = mydir + 'data/Turner_et_al/Breseq_Output_with_verification.txt'
        df_out = open(mydir + 'data/Tenaillon_et_al/gene_by_pop.csv', 'w')
        pop_by_gene_dict = {}
        for i, line in enumerate(open(df_in, 'rb')):
            line = line.decode(errors='ignore')
            line = line.strip().split('\t')
            treatment = line[0]
            sample = line[1]
            keep = line[12]
            gene = line[9]
            if keep == 'Y':
                gene = gene.replace('?', '')
                gene = gene.replace('[', '')
                gene = gene.replace(']', '')
                gene = gene.strip()

                # these are all "common names" of the genes used in this file
                # there is no way to map these names to the reference bc the
                # shortened names are not used in the annotation,
                # so I had to manually look up each gene and find its respective
                # old_locus_tag
                if (gene == 'GlcF'):
                    gene = 'Bcen2424_0732'
                if (gene == 'thiG'):
                    gene = 'Bcen2424_0412'
                if (gene == 'glcF'):
                    gene = 'Bcen2424_0732'
                if (gene == 'hemC'):
                    gene = 'Bcen2424_2425'
                if (gene == 'flgI'):
                    gene = 'Bcen2424_3018'
                if (gene == 'flgA'):
                    gene = 'Bcen2424_3026'

                # this old_locus_tag is not in the annotated reference assembly
                # GCF_000203955.1
                if gene == 'Bcen2424_5223':
                    continue

                if gene in gene_to_locus_tag:
                    locus_tag = gene_to_locus_tag[gene]
                else:
                    locus_tag = old_locus_tag_to_locus_tag[gene]

                treatment = treatment.replace(' ', '_')
                treatment = treatment.replace(',', '')
                treatment = treatment.replace('"', '')
                treatment = treatment.lower()

                treatment_strain = treatment + '_' + sample

                if '_small_bead_' in treatment_strain:
                    continue

                if locus_tag not in pop_by_gene_dict:
                    pop_by_gene_dict[locus_tag] = {}

                if treatment_strain in pop_by_gene_dict[locus_tag]:
                    pop_by_gene_dict[locus_tag][treatment_strain] += 1
                else:
                    pop_by_gene_dict[locus_tag][treatment_strain] = 1

        df = pd.DataFrame.from_dict(pop_by_gene_dict)
        df = df.fillna(0)
        # original data has "LIne" instead of "Line"
        df.index = df.index.str.replace('LIne', 'Line', regex=True)
        # remove rows and columns with all zeros
        df_out = mydir + 'data/Turner_et_al/gene_by_pop.txt'
        df.to_csv(df_out, sep = '\t', index = True)






class likelihood_matrix_array:
    def __init__(self, array, gene_list, dataset):
        self.array = np.copy(array)
        self.gene_list = gene_list
        self.dataset = dataset

    def get_gene_lengths(self, **keyword_parameters):
        if self.dataset == 'Good_et_al':
            conv_dict = good_et_al().parse_convergence_matrix(mydir + "/data/Good_et_al/gene_convergence_matrix.txt")
            length_dict = {}
            if ('gene_list' in keyword_parameters):
                for gene_name in keyword_parameters['gene_list']:
                    length_dict[gene_name] = conv_dict[gene_name]['length']
                #for gene_name, gene_data in conv_dict.items():
            else:
                for gene_name, gene_data in conv_dict.items():
                    length_dict[gene_name] = conv_dict[gene_name]['length']
            return(length_dict)

        elif self.dataset == 'Tenaillon_et_al':
            with open(mydir + '/data/Tenaillon_et_al/gene_size_dict.txt', 'rb') as handle:
                length_dict = pickle.loads(handle.read())
                if ('gene_list' in keyword_parameters):
                    return { gene_name: length_dict[gene_name] for gene_name in keyword_parameters['gene_list'] }
                else:
                    return(length_dict)

        elif self.dataset == 'Turner_et_al':
            with open(mydir + '/data/Turner_et_al/gene_size_dict.txt', 'rb') as handle:
                length_dict = pickle.loads(handle.read())
                if ('gene_list' in keyword_parameters):
                    return { gene_name: length_dict[gene_name] for gene_name in keyword_parameters['gene_list'] }
                else:
                    return(length_dict)

        elif self.dataset == 'Wannier_et_al':
            with open(mydir + '/data/Wannier_et_al/gene_size_dict.txt', 'rb') as handle:
                length_dict = pickle.loads(handle.read())
                if ('gene_list' in keyword_parameters):
                    return { gene_name: length_dict[gene_name] for gene_name in keyword_parameters['gene_list'] }
                else:
                    return(length_dict)


    def get_likelihood_matrix(self):
        genes_lengths = self.get_gene_lengths(gene_list = self.gene_list)
        #L_mean = np.mean(list(genes_lengths.values()))
        L_i = np.asarray(list(genes_lengths.values()))
        n_genes = np.count_nonzero(self.array, axis=1)
        n_tot = np.sum(self.array, axis=1)
        m_mean = np.true_divide(n_tot, n_genes)
        array_bin = (self.array > 0).astype(int)
        length_matrix = L_i*array_bin
        rel_length_matrix = length_matrix /np.true_divide(length_matrix.sum(1),(length_matrix!=0).sum(1))[:, np.newaxis]
        # length divided by mean length, so take the inverse
        with np.errstate(divide='ignore'):
            rel_length_matrix = (1 / rel_length_matrix)
        rel_length_matrix[rel_length_matrix == np.inf] = 0

        m_matrix = self.array * rel_length_matrix
        r_matrix = (m_matrix / m_mean[:,None])

        return r_matrix



# run clean_data first
wannier_et_al().get_gene_lengths()
wannier_et_al().clean_data()

good_et_al().reformat_convergence_matrix(mut_type = 'P')
good_et_al().reformat_convergence_matrix(mut_type = 'F')

tenaillon_et_al().clean_data()
tenaillon_et_al().pop_by_gene_tenaillon()


turner_et_al().clean_data()
