from __future__ import division
import os
import pickle
import operator
import sys
import random
import copy
from itertools import compress
import numpy as np
import pandas as pd
import parevol_tools as pt
import clean_data as cd

import matplotlib.pyplot as plt

from scipy import stats

df_non_path = pt.get_path() + "/data/Tenaillon_et_al/gene_by_pop_nonsyn.txt"
df_non = pd.read_csv(df_non_path, sep="\t", header="infer", index_col=0)
genes_non = df_non.columns.to_list()
df_non_np = df_non.values
df_non_np = np.transpose(df_non_np)

mean_all = []
var_all = []

for gene in df_non_np:

    if sum(gene>0) < 5:
        continue

    mean_all.append(np.mean(gene))
    var_all.append(np.var(gene))

mean_all = np.asarray(mean_all)
var_all = np.asarray(var_all)


slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(mean_all), np.log10(var_all))



slope_, intercept_, r_value_, p_value_, std_err_ = stats.linregress(np.log10(mean_all[mean_all<0.5]), np.log10(var_all[mean_all<0.5]))

print(slope)

print(slope_)

fig, ax = plt.subplots(figsize=(4,4))

ax.scatter(mean_all, var_all)


x_log10 = np.log10(np.logspace(-1.5, -0.1, num=1000, endpoint=True, base=10))

# hypothetical slope of 2
#ratio = (slope - 2) / std_err
#pval = stats.t.sf(np.abs(ratio), len(means)-2)*2

#ax.scatter(means, variances, c='#175ac6', marker = 'o', s = 110, \
#    edgecolors='none', linewidth = 0.6, alpha = 0.9, zorder=3)

ax.plot(10**x_log10, 10**(intercept + (x_log10 * slope)), ls='--', c='k', lw=2, label = r'$b = $' + str(round(slope/2, 3))  + ' OLS regression' )
ax.plot(10**x_log10, 10**(intercept + (x_log10 * slope_)), ls='--', c='grey', lw=2, label = r'$b = 1$' + ", Poisson")



ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)


fig.tight_layout()
fig.savefig(pt.get_path() + '/figs/mean_vs_var.png', format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
