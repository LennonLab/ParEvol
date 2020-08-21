from __future__ import division
import os, pickle, operator
import random
from itertools import compress
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from sklearn.metrics.pairwise import euclidean_distances
#from asa159 import rcont2
from scipy import linalg as LA

from sklearn.decomposition import PCA

import clean_data as cd

df_path = '/Users/WRShoemaker/GitHub/ParEvol/data/Tenaillon_et_al/gene_by_pop.txt'
#df_path = mydir + '/data/Tenaillon_et_al/gene_by_pop.txt'
df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
df_np = df.values
gene_names = df.columns.values
df_np_delta = cd.likelihood_matrix_array(df_np, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()

X = df_np_delta/df_np_delta.sum(axis=1)[:,None]
X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
#X = X - np.mean(X, axis = 0)

pca = PCA()
pca_X = pca.fit(X)
