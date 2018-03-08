from __future__ import division
import os
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.misc import factorial


mydir = os.path.expanduser("~/GitHub/ParEvol/")



def plot_interactions():
    # http://compsysbio.org/bacteriome/download.php
    df = pd.read_csv(mydir + 'data/functional_interactions.txt' , sep = '\t', header = None)
    df.columns = ['protein1', 'protein2', 'strength']
    protein_set = set(df.protein1.tolist()) & set(df.protein1.tolist())
    ## get set.....
    df = pd.crosstab(df.protein1, df.protein2)
    idx = df.columns.union(df.index)
    df = df.reindex(index = idx, columns=idx, fill_value=0)
    interactions = df.sum(axis=1).values
    #interactions = [x for x in interactions if x > 0]

    def poisson(k, lamb):
        """poisson pdf, parameter lamb is the fit parameter"""
        return (lamb**k/factorial(k)) * np.exp(-lamb)


    def negLogLikelihood(params, data):
        """ the negative log-Likelohood-Function"""
        lnl = - np.sum(np.log(poisson(data, params[0])))
        return lnl

    result = minimize(negLogLikelihood,  # function to minimize
                  x0=np.ones(1),     # start value
                  args=(interactions,),      # additional arguments for function
                  method='Powell',   # minimization method, see docs
                  )

    fig = plt.figure()
    plt.hist(interactions, normed=True, bins=30)

    x=np.linspace(0,14,200)
    plt.plot(x, poisson(x, result.x), 'r-', lw=2)

    plt.xlabel("Number of interactions for a given protein" , fontsize = 12)
    plt.ylabel('Frequency', fontsize = 11)
    fig_name = mydir + 'figs/ppi_edges.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()
    #for p in protein_set:



plot_interactions()

# start with one gene, the NK model

def int2bits(k, N):
    x = list(map(int, bin(k)[2:]))
    pad = N - len(x)
    x = [0]*pad + x
    return x

def all_genotypes(N):
    return np.array([int2bits(k, N) for k in range(2**N)], dtype=bool)

N = 4
K = 1

genotypes = all_genotypes(N)
#print(genotypes)
#print(list(map(int, bin(5)[2:])))

# contribs = a dictionary of the K neighbors that each site interacts with

contribs = {
        i: sorted(np.random.choice(
            [n for n in range(N) if n != i],
            K,
            replace=False
        ).tolist() + [i])
        for i in range(N)
    }

#print(contribs)
