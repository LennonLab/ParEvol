from __future__ import division
import math, random, itertools
import numpy as np
import pandas as pd
import scipy.stats as stats
import parevol_tools as pt


df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
