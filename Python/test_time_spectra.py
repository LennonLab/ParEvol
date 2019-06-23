from __future__ import division
import os, sys
import numpy as np
import pandas as pd
import parevol_tools as pt
import clean_data as cd

df_path = os.path.expanduser("~/GitHub/ParEvol/")  + '/data/Good_et_al/gene_by_pop.txt'
df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
to_exclude = pt.complete_nonmutator_lines()
to_exclude.append('p5')
df = df[df.index.str.contains('|'.join(to_exclude))]

X = cd.likelihood_matrix_array(df, df.columns.values, 'Good_et_al').get_likelihood_matrix()
X_df = pd.DataFrame(data=X, index=df.index.values, columns=df.columns.values)

X_df_p5 = X_df[X_df.index.str.contains('m5_')]
X_df_p5 = X_df_p5.loc[:, (X_df_p5 != 0).any(axis=0)]
X_p5 = X_df_p5.values

print(X_df_p5)
