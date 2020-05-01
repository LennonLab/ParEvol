from __future__ import division
import os, pickle, operator
import random
from itertools import compress
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from sklearn.metrics.pairwise import euclidean_distances
from asa159 import rcont2
from scipy import linalg as LA

from sklearn.decomposition import PCA
import sys

N=int(sys.argv[1])

#import time
#start_time = time.time()



#Running Python on Carbonate

#module avail python
#module unload python/2.7.13

#module unload python/3.6.1
#module switch python/2.7.13 python/3.6.1

#module load anaconda/python3.6/4.3.1

#conda create -n ParEvol python=3.6



mydir = '/N/dc2/projects/Lennon_Sequences/ParEvol'
#mydir = '/Users/WRShoemaker/GitHub/ParEvol'

def get_random_matrix(c_in):
    #```GNU Lesser General Public License v3.0 code from https://github.com/maclandrol/FisherExact```
    # f2py -c -m asa159 asa159.f90
    #c = array
    # remove empty columns
    empty_cols = (np.where(~c_in.any(axis=0))[0])
    empty_rows = (np.where(~c_in.any(axis=1))[0])
    c = np.delete(c_in, empty_cols, axis=1)
    c = np.delete(c, empty_rows, axis=0)

    key = np.array([False], dtype=bool)
    ierror = np.array([0], dtype=np.int32)
    sr, sc = c.sum(axis=1).astype(np.int32), c.sum(axis=0).astype(np.int32)
    nr, nc = len(sr), len(sc)
    n = np.sum(sr)
    replicate=1000
    results = np.zeros(replicate)

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
            seed = np.array([seed], dtype=np.int32)
        except:
            try:
                import time
                seed = int(time.time())
                seed = np.array([seed], dtype=np.int32)
            except:
                seed = 12345
                seed = np.array([seed], dtype=np.int32)

    if n < wkslimit:
        # we can just set the limit  to the table sum
        wkslimit = n
        pass
    else:
        # throw error immediately
        raise ValueError(
            "Limit of %d on the table sum exceded (%d), please increase workspace !" % (DFAULT_MAX_TOT, n))

    maxtot = np.array([wkslimit], dtype=np.int32)
    fact = np.zeros(wkslimit + 1, dtype=np.float, order='F')
    observed = np.zeros((nr, nc), dtype=np.int32, order='F')

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
        #    np.insert(tmp_observed, empty_column, nr, axis=1)
        rndm_matrix = np.reshape(tmp_observed, (nr,nc))
        for empty_column in empty_cols:
            rndm_matrix = np.insert(rndm_matrix, empty_column, 0, axis=1)
        for empty_row in empty_rows:
            rndm_matrix = np.insert(rndm_matrix, empty_row, 0, axis=0)

        return rndm_matrix
        #return np.reshape(tmp_observed, (nr,nc))




def get_mean_pairwise_euc_distance(array, k = 3):
    #X = array[0:k,:]
    X = array[:,:k]
    row_sum = np.sum( euclidean_distances(X, X), axis =1)
    return sum(row_sum) / ( len(row_sum) * (len(row_sum) -1)  )




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
        conv_dict = self.parse_convergence_matrix(mydir + "/data/Good_et_al/gene_convergence_matrix.txt")
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
            df_out = mydir + '/data/Good_et_al/gene_by_pop.txt'
            #df_delta_out = mydir + 'data/Good_et_al/gene_by_pop_delta.txt'
        elif mut_type == 'P':
            df_out = mydir + '/data/Good_et_al/gene_by_pop_poly.txt'
            #df_delta_out = mydir + 'data/Good_et_al/gene_by_pop_poly_delta.txt'
        else:
            print("Argument mut_type not recognized")
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





def rndm_sample_tenaillon(N, k_eval=3, iter1=1000, iter2=10000, sample_bs = 100, iter_bs=10000):
    df_path = mydir + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    n_rows = df_np.shape[0]
    df_out = open(mydir + '/data/Tenaillon_et_al/power_sample_size_' + str(N) + '.txt', 'w')
    df_out.write('\t'.join(['N', 'G', 'Power', 'Power_025', 'Power_975', 'z_score_mean', 'z_score_025', 'z_score_975']) + '\n')
    #Ns = [20]
    #Ns = list(range(20, n_rows, 5))
    #for N in Ns:
    p_values = []
    z_scores = []
    G_list = []
    for i in range(iter1):
        df_np_i = df_np[np.random.choice(n_rows, size=N, replace=False, p=None), :]
        gene_bool = np.all(df_np_i == 0, axis=0)
        # flip around to select gene_size
        gene_names_i = list(compress(gene_names, list(map(operator.not_, gene_bool))))
        G_list.append(len(gene_names_i))
        df_np_i = df_np_i[:,~np.all(df_np_i == 0, axis=0)]
        np.seterr(divide='ignore')
        df_np_i_delta = likelihood_matrix_array(df_np_i, gene_names_i, 'Tenaillon_et_al').get_likelihood_matrix()
        X = df_np_i_delta/df_np_i_delta.sum(axis=1)[:,None]
        X -= np.mean(X, axis = 0)
        pca = PCA()
        pca_X = pca.fit_transform(X)
        mpd = get_mean_pairwise_euc_distance(pca_X, k=k_eval)
        mpd_null = []
        for j in range(iter2):
            df_np_i_j = get_random_matrix(df_np_i)
            np.seterr(divide='ignore')
            df_np_i_j_delta = likelihood_matrix_array(df_np_i_j, gene_names_i, 'Tenaillon_et_al').get_likelihood_matrix()
            X_j = df_np_i_j_delta/df_np_i_j_delta.sum(axis=1)[:,None]
            X_j -= np.mean(X_j, axis = 0)
            pca_X_j = pca.fit_transform(X_j)
            mpd_null.append(get_mean_pairwise_euc_distance(pca_X_j, k=k_eval))
        p_values.append(len( [m for m in mpd_null if m > mpd] ) / len(mpd_null))
        z_scores.append( (mpd - np.mean(mpd_null)) / np.std(mpd_null) )

    power = len([n for n in p_values if n < 0.05]) / len(p_values)
    #print(p_values)
    power_bootstrap = []
    for p in range(iter_bs):
        p_values_sample = random.sample(p_values, sample_bs)
        power_sample = len([n for n in p_values_sample if n < 0.05]) / len(p_values_sample)
        power_bootstrap.append(power_sample)
    power_bootstrap.sort()

    z_scores_bootstrap = []
    for p in range(iter_bs):
        z_scores_bootstrap.append(np.mean( random.sample(z_scores, sample_bs)  ))
    z_scores_bootstrap.sort()


    # return number of genes, power, power lower, power upper
    #return  power, power_bootstrap[int(10000*0.025)], power_bootstrap[int(10000*0.975)]
    df_out.write('\t'.join([str(N), str(np.mean(G_list)), str(power), str(power_bootstrap[int(iter_bs*0.025)]), str(power_bootstrap[int(iter_bs*0.975)]), str(np.mean(z_scores)), str(z_scores_bootstrap[int(iter_bs*0.025)]), str(z_scores_bootstrap[int(iter_bs*0.975)])   ]) + '\n')
    df_out.close()



rndm_sample_tenaillon(N)


#print("--- %s seconds ---" % (time.time() - start_time))
