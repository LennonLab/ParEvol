from __future__ import division
import numpy as np
import pandas as pd
import parevol_tools as pt
from functools import reduce

t0 = {   'Gene1': [0, 1, 0],
         'Gene2': [2, 0, 3],
         'Gene3': [0, 0, 1]}

t1 = {   'Gene1': [1, 1, 0],
         'Gene2': [3, 0, 3],
         'Gene3': [0, 2, 2]}

t2 = {   'Gene1': [2, 1, 0],
         'Gene2': [4, 0, 3],
         'Gene3': [0, 2, 3]}

df_t0 = pd.DataFrame(t0, index = ['p1_t0', 'p2_t0', 'p3_t0'])
df_t1 = pd.DataFrame(t1, index = ['p1_t1', 'p2_t1', 'p3_t1'])
df_t2 = pd.DataFrame(t2, index = ['p1_t2', 'p2_t2', 'p3_t2'])
dfs = [df_t0, df_t1, df_t2]

df = df_t0.append([df_t1, df_t2])

df_0 = df[df.index.str.contains('_t0')].as_matrix()

time_points = 3
for t in range(1, time_points):
    df_t_minus1 = df[df.index.str.contains('_t' + str(t-1))]
    df_t = df[df.index.str.contains('_t' + str(t))]
    df_diff = df_t.as_matrix() - df_t_minus1.as_matrix()

    #print(np.sum(df_t.as_matrix(), axis=1))
    #df_t_rndm = df_t_minus1 + pt.random_matrix(df_diff)
    df_0 = df_0 + pt.random_matrix(df_diff)
    #print(np.sum(df_t_rndm.as_matrix(), axis=1))

    #print(np.sum(df_t.as_matrix(), axis=0))
    #print(np.sum(df_t_rndm.as_matrix(), axis=0))


print(np.sum(df_0, axis=1))
print(np.sum(df[df.index.str.contains('_t2')].as_matrix(), axis=1))


print(np.sum(df_0, axis=0))
print(np.sum(df[df.index.str.contains('_t2')].as_matrix(), axis=0))
