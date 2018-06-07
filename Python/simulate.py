from __future__ import division
import os, itertools, math
import numpy as np
import parevol_tools as pt
#from sympy.utilities.iterables import multiset_permutations

def p_fix_wm(N, s):
    # function for probability of fixation under weak selection
    return( (2*s)  / (1 - np.exp(-2*N*s)))

def dfe_0_sample(s):
    return np.random.exponential(abs(s))

def dfe_eq_sample(s, t, dfe_o_param):
    return dfe_0_sample(s) * ((1 + np.exp(2*N*s) ) ** -1)

#def dfe_t(N, U_b, s, L_b, t):
    #np.rand


class simulate_NK:
    def __init__(self, N, K, G, alpha):
        self.N = N
        self.K = K
        self.G = G
        self.alpha = alpha


    def neighbors(self, genotype, genotypes, only_previous = False):
        if only_previous == True:
            # keep only the previous neighbor
            return [g for g in genotypes if (abs(g ^ genotype).sum() == 1) and (np.sum(g) < np.sum(genotype))]
        else:
            return [g for g in genotypes if abs(g ^ genotype).sum() == 1]


    def int2bits(self, k):
        x = list(map(int, bin(k)[2:]))
        pad = self.N - len(x)
        x = [0]*pad + x
        return x


    def bits2int(self, bits_list):
        #out_list = []
        #for item in bits_list:
        #    out_list.append(''.join([str(int(i == True))  for i in item]))
        #return out_list
        return ''.join([str(int(i == True))  for i in bits_list])


    def all_genotypes(self):
        return np.array([self.int2bits(k) for k in range(2**self.N)], dtype=bool)


    def fitness_i(self, genotype, i, contribs, mem):
        key = tuple(zip(contribs[i], genotype[contribs[i]]))
        if key not in mem:
            mem[key] = np.random.uniform(0, 1)
        return mem[key]


    def fitness(self, genotype, contribs, mem):
        return np.mean([
            self.fitness_i(genotype, i, contribs, mem) # ω_i
            for i in range(len(genotype))
        ])


    def get_maxima(self):
        if self.N % self.G != 0:
            raise Exception('N divided by G does not return a whole number')
        max_genotypes = []
        genotypes = self.all_genotypes()
        contribs = {
            i: sorted(np.random.choice(
                [n for n in range(self.N) if n != i],
                self.K,
                replace=False
            ).tolist() + [i])
            for i in range(self.N)
        }

        # get dict describing what sites are in what gene
        gene_dict = {}
        if self.alpha == float(0):
            genes = np.random.choice(self.N, size=[self.G, int(self.N/self.G)], replace=False)
        else:
            # weighted probability. A value of alpha
            print(contribs)
            p_null = [1/self.N] * self.N
            #print(list(zip(, p_null)))
        #print(contribs)

        #for i, gene in enumerate(genes):
        #    for site in gene:
        #        gene_dict[site] = i
        #else:

        #print(gene_dict)
        fitness_mem = {}
        ws = [self.fitness(g, contribs, fitness_mem) for g in genotypes]
        max_w = max(ws)
        min_w = min(ws)
        print(fitness_mem)
        #print(bits2int(genotypes[ws.index(max_w)]))
        for i, genotype in enumerate(genotypes):
            wi = (self.fitness(genotype, contribs, fitness_mem) - min_w) / (max_w - min_w)
            maximum = True
            minimum = True
            for g in self.neighbors(genotype, genotypes):
                w = (self.fitness(g, contribs, fitness_mem) - min_w) / (max_w - min_w)
                if w > wi:
                    maximum = False
                if w < wi:
                    minimum = False
            if maximum:
                max_genotypes.append(genotype)

        return max_genotypes





print(simulate_NK(4, 1, 2, 0.1).get_maxima())



def fun(dct, value, path=()):
    for key, val in dct.items():
        if val == value:
            yield path + (key, )
    for key, lst in dct.items():
        if isinstance(lst, list):
            for item in lst:
                for pth in fun(item, value, path + (key, )):
                    yield pth


# total number of trajectories = math.factorial(L)
#def test_parse_genotype():
#    path_dict = {}
#    for i, genotype in enumerate(genotypes):
#        wi = (fitness(genotype, contribs, fitness_mem) - min_w) / (max_w - min_w)
#        maximum = True
#        minimum = True
#        for g in neighbors(genotype, genotypes):
#            # keep the previous neighbor, g
#            w = (fitness(g, contribs, fitness_mem) - min_w) / (max_w - min_w)
#            w_delta = wi - w
#            if wi - w < 0:
#                continue
#            gt_int = bits2int(genotype)
#            if gt_int.count('1') == 1:
#                path_dict[gt_int] = {}
#            else:
#                #print(gen_dict_extract(bits2int(g), path_dict))
#                if bits2int(g) in path_dict:
#                    path_dict[bits2int(g)] = gt_int
#                key_generator = fun(path_dict, bits2int(g))
#                key_list = []
#                for item in key_generator:
#                    key_list.append(item[0])
#                #print(key_list)
#                #if key_list == None:
#                #    continue
#                print(bits2int(g), list(key_list), len((key_list)))
#                #if key_list == None:
#                #    continue
#                #print(key_list, bits2int(g))

#                # figure out what to with the generator
#                # use the list of keys to add the new value to the dictioary
#                # then parse your list

#                ##### but firstttttt, test the fun function on a nested dictioary

#    print(path_dict)
