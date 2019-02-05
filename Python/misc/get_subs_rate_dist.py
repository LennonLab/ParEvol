from __future__ import division
import math, random, itertools, os, pickle
import numpy as np
import pandas as pd
import parevol_tools as pt
import matplotlib.pyplot as plt

df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
gene_counts = df.sum(axis=0)

with open(pt.get_path() + '/data/Tenaillon_et_al/gene_size_dict.txt', 'rb') as handle:
    length_dict = pickle.loads(handle.read())

rates = []
for i, j in gene_counts.iteritems():
    print(i, j)
    rates.append(j / length_dict[i] )

print(rates)


fig = plt.figure()
plt.hist(rates, bins=70,  alpha=0.8, color = '#175ac6')

# it's overdispersed

plt.tight_layout()
fig_name = pt.get_path() + '/figs/sub_hist.png'
fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
