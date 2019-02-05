from __future__ import division
import os, re
import parevol_tools as pt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_good_path = pt.get_path() + '/data/Good_et_al/gene_by_pop.txt'
df_good =  pd.read_csv(df_good_path, sep = '\t', header = 'infer', index_col = 0)
samples =  [ int(x.split('_')[1]) for x in df_good.index.values]
#print(str(max(samples)))

df_good_62750 = df_good[df_good.index.str.contains(str(max(samples)))]
to_exclude = pt.complete_nonmutator_lines()
to_exclude.append('p5')

df_good_62750_nomut = df_good_62750[df_good_62750.index.str.contains('|'.join( to_exclude))]
df_good_62750_nomut_sum = df_good_62750_nomut.sum(axis = 0)

df_good_62750_nomut_sum_delta = pt.likelihood_matrix(df_good_62750_nomut_sum.to_frame().T, 'Good_et_al').get_likelihood_matrix()

mult_list = df_good_62750_nomut_sum_delta.loc[0].tolist()
mult_list = list(filter((float(0)).__ne__, mult_list))

fig = plt.figure()
plt.hist(mult_list,bins=50, weights=np.zeros_like(mult_list) + 1. / len(mult_list), alpha=0.8, color = '#175ac6')
plt.xlabel('Gene multiplicity')
plt.ylabel('Frequency')
print("mean = " + str(np.mean(mult_list)))
print("mean = " + str(np.var(mult_list)))

# it's overdispersed

plt.tight_layout()
fig_name = pt.get_path() + '/figs/mult_hist.png'
fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



#print(df_good_62750_nomut.sum(axis = 0).tolist())
#print(df_good_62750_nomut_sum)
count_list = df_good_62750_nomut.sum(axis = 0).tolist()
count_list = list(filter((float(0)).__ne__, count_list))

fig = plt.figure()
plt.hist(count_list,bins=50, weights=np.zeros_like(count_list) + 1. / len(count_list), alpha=0.8, color = '#175ac6')
plt.xlabel('Number of mutations in a gene', fontsize = 24)
plt.ylabel('Frequency', fontsize = 24)
print("mean = " + str(np.mean(count_list)))
print("mean = " + str(np.var(count_list)))

# it's overdispersed

plt.tight_layout()
fig_name = pt.get_path() + '/figs/count_hist.png'
fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()




#### do the same for tenaillon et al
