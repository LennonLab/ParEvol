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


def run_sim():
    N = 1000
    U = 0.001






def neighbors(genotype, genotypes):
    return [g for g in genotypes if abs(g ^ genotype).sum() == 1]

def int2bits(k, N):
    x = list(map(int, bin(k)[2:]))
    pad = N - len(x)
    x = [0]*pad + x
    return x

def bits2int(bits_list):
    out_list = []
    for item in bits_list:
        out_list.append(''.join([str(int(i == True))  for i in item]))
    return out_list

def all_genotypes(N):
    return np.array([int2bits(k, N) for k in range(2**N)], dtype=bool)

def fitness_i(genotype, i, contribs, mem):
    key = tuple(zip(contribs[i], genotype[contribs[i]]))
    if key not in mem:
        mem[key] = np.random.uniform(0, 1)
    return mem[key]

def fitness(genotype, contribs, mem):
    return np.mean([
        fitness_i(genotype, i, contribs, mem) # ω_i
        for i in range(len(genotype))
    ])


G = 2
N = 4
K = 1
genotypes = all_genotypes(N)
contribs = {
        i: sorted(np.random.choice(
            [n for n in range(N) if n != i],
            K,
            replace=False
        ).tolist() + [i])
        for i in range(N)
    }
fitness_mem = {}
ws = [fitness(g, contribs, fitness_mem) for g in genotypes]
# adaptive walk
fitness_dict = dict(zip(bits2int(genotypes), ws))

# find fitness maximum, get sequence of maximum, construct binary tree of possibilies,
# find paths that actually increase fitness

max_w = max(ws)
max_genotype = bits2int(genotypes)[ws.index(max_w)]
steps = max_genotype.count('1')

# total number of trajectories = math.factorial(L)

test = '1111'



def get_paths(test):
    start = '0000'
    steps = test.count('1')
    for step in range(1, 2):
        to_permute = ([True] * step) + ([False] * (steps-step))
        step_combos = list(pt.perm_unique(to_permute))
        for step_combo in step_combos:
            print(step_combo)
            genoty = ''.join([str(int(i == True))  for i in step_combo])
            print(fitness_dict)
            w = fitness_dict[genoty]
            print(w)
        #print(tttttt)



#print(max_genotype)
get_paths(test)





def climb_hill(fitness_dict, N):
    init = '0' * N
    fitness = fitness_dict[init]

    print(fitness)

    #while

#climb_hill(fitness_dict, N)
#print(get_options('000000', N, 0))
#print(genotypes)
#print(ws)
