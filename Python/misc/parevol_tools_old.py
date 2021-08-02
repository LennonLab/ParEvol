from __future__ import division
import os, pickle, math, random, itertools
from itertools import combinations
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA

from scipy.linalg import block_diag
from scipy.special import comb
import scipy.stats as stats
from scipy.special import gammaln
from scipy import linalg as LA

#import networkx as nx
from asa159 import rcont2
from copy import copy
import matplotlib.colors as cls





# code from https://github.com/lubeme/Scale-Free-Random-Walks-Networks
def random_walks_powerlaw_cluster_graph(m,n,cc,seed=None):
    """Return random graph using Herrera-Zufiria random walks model.

    A Scale-free graph of n nodes is grown by attaching new nodes
    each with m edges that are connected to existing by performing
    random walks,using only local information of the network.
    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    cc: int
        clustering control parameter, from 0 to 100. Increasing control
        parameter implies increasing the clustering of the graph
    seed : int, optional
        Seed for random number generator (default=None).
    Returns
    -------
    G : Graph

    Notes
    -----
    The initialization is a circular graph with an odd number of nodes.
    For small values of m initial graph has at least 11 nodes.
    References
    ----------
    .. [1] Herrera, C.; Zufiria, P.J.; , "Generating scale-free networks
    with adjustable clustering coefficient via random walks,"
    Network Science Workshop (NSW),
    2011 IEEE , vol., no., pp.167-172, 22-24 June 2011
    """


    if m < 1 or  m >=n or cc < 0 or cc > 100:
        raise nx.NetworkXError(\
            "The network must have m>=1, m<n and "
            "cc between 0 and 100. m=%d,n=%d cc=%d"%(m,n,cc))

    if seed is not None:
        random.seed(seed)

    nCero= max(11,m)
    if nCero%2==0:
        nCero+=1
    #initialise graph
    G= nx.generators.classic.cycle_graph(nCero)
    G.name="Powerlaw-Cluster Random-Walk Graph"

    #list of probabilities 'pi' associated to each node
    #representing a genetic factor
    p=stats.bernoulli.rvs(cc/float(100),size=n)

    #main loop of the algorithm
    for j in range(nCero,n):
        #Choose Random node
        vs =random.randrange(0,G.number_of_nodes())
        #random walk of length>1 beginning on vs
        l = 7
        ve=vs
        for i in range(l):
            neighborsVe = G.neighbors(ve)
            neighborsVe_list = list(neighborsVe)
            # random.choice(numberList)
            #ve= list(neighborsVe)[random.randrange(0,len(list(neighborsVe)))]
            ve = random.choice(neighborsVe_list)
            #len(list(somegraph.neighbors(somenode)))


        markedVertices=[]
        #mark ve
        markedVertices.append(ve)
        vl=ve
        #Mark m nodes
        for i in range(m-1):
            #Random walk of l = [1 , 2] depending on the
            #genetic factor of the node vl
            l =2 - p[vl]
            vll=vl
            #Random Walk starting on vl, avoiding already marked vertices
            while ((vll in markedVertices)):
                print(vll)
                for k in range(l):
                    neighborsVl = G.neighbors(vll)
                    neighborsVl_list = list(neighborsVl)
                    #vll= neighborsVl[random.randrange(0,len(neighborsVl))]
                    vll = random.choice(neighborsVl_list)
            vl=vll
            #mark vl
            markedVertices.append(vl)
        #Add the new node
        G.add_node(j)
        #Assign the node a pi
        #Add the m marked neighbors to vl
        for i in range(m):
            G.add_edge(j,markedVertices[i])
    return G
