from __future__ import division
import math, random, itertools
from itertools import combinations
from scipy.special import comb

def comb_n_muts_k_genes(n, gene_sizes):
    #n = 7
    #gene_sizes = [2, 3, 4]
    k = len(gene_sizes)
    def findsubsets(S,m):
        return set(itertools.combinations(S, m))
    B = []
    for count in range(0, len(gene_sizes) + 1):
        #print(findsubsets(set(gene_sizes), count))
        for subset in findsubsets(set(gene_sizes), count):
            B.append(list(subset))
    number_ways = 0
    #print(B)
    for S in B:
        #print(sum(S), len(S))
        n_S = n + k - 1 - (sum(S) + (1 * len(S) ) )
        if n_S < (k-1):
            continue

        number_ways +=  ((-1) ** len(S)) * comb(N = n_S, k = k-1)
    return number_ways


def comb_n_muts_k_genes_loop(n, gene_sizes):
    k = len(gene_sizes)
    number_ways = 0
    def findsubsets(S,m):
        return set(itertools.combinations(S, m))
    B = []
    for count in range(0, len(gene_sizes) + 1):
        for subset in findsubsets(set(gene_sizes), count):
            #print( count, subset)
            subset = list(subset)
            n_S = n + k - 1 - (sum(subset) + (1 * len(subset) ) )
            if n_S < (k-1):
                continue
            number_ways +=  ((-1) ** len(subset)) * comb(N = n_S, k = k-1)
    return number_ways


def comb_n_muts_k_genes(k, gene_sizes):
    G = len(gene_sizes)
    gene_sizes = list(gene_sizes)
    number_states = 0
    for i in range(0, len(gene_sizes) + 1):
        comb_sum = 0
        for j in list(itertools.combinations(gene_sizes, i)):
            if (len(j) > 0): #and (len(j) < G):
                s_i_j = sum( j ) + (len(j)*1)
            else:
                s_i_j = sum( j )

            comb_s_i_j = comb(N = G+k-1-s_i_j, k = G-1)
            comb_sum += comb_s_i_j

        number_states += ((-1) ** i) * comb_sum

    print(number_states)



#print(comb_n_muts_k_genes(3, [3,4, 2]))
#print(comb_n_muts_k_genes_loop(20, [5,5, 5, 5]))

#comb_n_muts_k_genes_test(14, [3, 6, 9, 12, 15])

comb_n_muts_k_genes_test(2, [2,2])
