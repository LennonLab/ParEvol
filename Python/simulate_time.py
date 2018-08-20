from __future__ import division
import os, itertools, math
import numpy as np
import parevol_tools as pt
from scipy.integrate import quad
#from sympy.utilities.iterables import multiset_permutations


#def p_fix_wm(N, s):
#    # function for probability of fixation under weak selection
#    return( (2*s)  / (1 - np.exp(-2*N*s)))

#def dfe_0_sample(s):
#    return np.random.exponential(abs(s))

#def dfe_eq_sample(s, t, dfe_o_param):
#    return dfe_0_sample(s) * ((1 + np.exp(2*N*s) ) ** -1)

#def dfe_t(N, U_b, s, L_b, t):
    #np.rand


class simulate_time:

    def __init__(self, N, U_b, U_n, L, G, mean_scale = 0.01):
        self.N = N
        self.U_b = U_b
        self.U_n = U_n
        self.L = L
        self.G = G
        self.mean_scale = mean_scale

    def sample_exp(self, scale, size = 1):
        return np.random.exponential(abs(scale), size)

    def evolve(self):
        #gene_size_dict = {g:(self.L/self.G) for g in range(self.G)}
        #dfe_shape_dict = {g:self.sample_exp(self.mean_scale) for g in range(self.G)}
        sub_rate = self.N * self.U_b * 2 * self.sample_exp(self.mean_scale)[0]
        num_muts = np.random.poisson(sub_rate)
        return num_muts



print(simulate_time(10000, 0.001, 0.01, 1000, 10).evolve())
