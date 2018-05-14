from __future__ import division
import os
import numpy as np

mydir = os.path.expanduser("~/GitHub/ParEvol/")



def p_fix_wm(N, s):
    # function for probability of fixation under weak selection
    return( (2*s)  / (1 - np.exp(-2*N*s)))

def dfe_0_sample(s):
    return np.random.exponential(abs(s))

def dfe_eq_sample(s, N):
    return dfe_0_sample(s) * ((1 + np.exp(2*N*s) ) ** -1)

#def dfe_t(N, U_b, s, L_b, t):
    #np.rand



def run_sim():
    N = 1000
    U = 0.001
