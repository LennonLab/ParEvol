from __future__ import division
import os, pickle, math, random, itertools, re
from itertools import combinations
import numpy as np

import scipy.stats as stats

from scipy.special import iv

import statsmodels.stats.multitest as mt







def skellam_pdf(delta_n, lambda_1, lambda_2):

    if delta_n > 0:

        pmf = ((lambda_1/lambda_2)**(delta_n/2)) * iv(delta_n, 2*np.sqrt(lambda_1*lambda_2))
        pmf += ((lambda_2/lambda_1)**(delta_n/2)) * iv(-1*delta_n, 2*np.sqrt(lambda_1*lambda_2))
        pmf *= np.exp((-1*lambda_1) + (-1*lambda_2))

    else:

        pmf = np.exp((-1*lambda_1) + (-1*lambda_2)) * iv(0, 2*np.sqrt(lambda_1*lambda_2))

    return pmf



def calculate_survival(n_min = 2, alpha = 0.05):

    counts_1 = np.asarray([8,3,4,0,0,0,2,2,7,2,4,2,0,2,2,0,2,2,2,0,4,2,0,2,2,2,0,2,2,0,2,2,2,0,2,2,0,2,2,2,0,2,2,0,2,2,0,2,2,2,0,2,2])
    counts_2 = np.asarray([0,8,2,0,1,0,3,2,0,2,2,2,0,2,2,0,2,2,2,0,2,2,3,2,2,2,0,2,2,0,2,2,2,0,2,2,0,2,2,2,0,2,2,0,2,2,0,2,2,2,0,2,2])

    lambda_1 = sum(counts_1)/len(counts_1)
    lambda_2 = sum(counts_2)/len(counts_2)

    delta_n = np.absolute(counts_1-counts_2)

    delta_n_range = list(range(0,1000))
    delta_n_range_array = np.asarray(delta_n_range)

    delta_n_range_array_subset = delta_n_range_array[delta_n_range_array<=max(delta_n)]

    pmf = [skellam_pdf(i, lambda_1, lambda_2) for i in delta_n_range]
    pmf = np.asarray(pmf)

    survival_null = [ 1-sum(pmf[:i]) for i in range(len(pmf)) ]
    survival_null = np.asarray(survival_null)
    survival_null = survival_null[delta_n_range_array<=max(delta_n)]


    survival_obs = [ len(delta_n[delta_n>=i])/len(delta_n_range_array) for i in delta_n_range]
    survival_obs = np.asarray(survival_obs)
    survival_obs = survival_obs[delta_n_range_array<=max(delta_n)]

    P_values = [sum(pmf[delta_n_range.index(delta_n_i):]) for delta_n_i in delta_n]
    P_values = np.asarray(P_values)

    expected_number_genes = 0
    P_range = np.linspace(10**-4, 0.05, num=10000)[::-1]


    N_bar_P_star_div_N_P_star = []
    P_stars = []

    for P_range_i in P_range:

        N_P_star = len(P_values[P_values<P_range_i])
        N_bar_P_star = 0

        if N_P_star == 0:
            continue

        #for g in range(len(counts_1)):

        for delta_n_j_idx, delta_n_j in enumerate(delta_n):
            if delta_n_j < n_min:
                continue

            P_delta_n_j = sum(pmf[delta_n_range.index(delta_n_j):])

            if P_range_i > P_delta_n_j:
                # no gene specific indices so just multiply the final probability by number of genes
                N_bar_P_star += skellam_pdf(delta_n_j, lambda_1, lambda_2) * len(counts_1)


        N_bar_P_star_div_N_P_star.append(N_bar_P_star/N_P_star)
        P_stars.append(P_range_i)

    N_bar_P_star_div_N_P_star = np.asarray(N_bar_P_star_div_N_P_star)
    P_stars = np.asarray(P_stars)

    position_P_star = np.argmax(N_bar_P_star_div_N_P_star<=0.05)

    P_star = P_stars[position_P_star]

    #delta_n_P = delta_n


    print(len(delta_n), len(P_values))


    return delta_n_range_array_subset, survival_obs, survival_null, P_values, P_star





calculate_survival()
