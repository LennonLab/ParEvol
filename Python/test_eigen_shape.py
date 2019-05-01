from __future__ import division
import clean_data as cd
import os
import pandas as pd
import parevol_tools as pt
import numpy as np
import matplotlib.pyplot as plt


df_path = os.path.expanduser("~/GitHub/ParEvol") + '/data/Tenaillon_et_al/gene_by_pop.txt'
df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
df_np = df.values
#print(df.columns.values)
df_np_delta = cd.likelihood_matrix_array(df_np, df.columns.values,'Tenaillon_et_al').get_likelihood_matrix()
X = pt.get_mean_center(df_np_delta)
cov = (1 / (X.shape[0])) * X @ np.transpose(X)

ev , eig = np.linalg.eig(cov)
#ev[::-1].sort()
ev.sort()
ev = ev[1:]



def get_ev_spacings(array):
    spacings = []
    for i in range(len(array) -1):
        s_i = array[i+1] - array[i]
        spacings.append(s_i)
    spacings_np = np.asarray(spacings)
    spacings_np = spacings_np / np.mean(spacings_np)
    return spacings_np



s_poisson = np.linspace(0.002, 30, num=1000)
P_s_poisson = np.exp(-1* s_poisson)

spacings = get_ev_spacings(ev)
print(spacings)
print(P_s_poisson)

# kde or just plot poisson?


fig = plt.figure()
plt.plot(s_poisson, P_s_poisson)
plt.hist(spacings, density=True, bins=30)
#plt.hist(np.log10(spacings), bins=30, histtype='stepfilled', normed=True, alpha=0.6, color='b')
plt.xlabel("Eigenvalue spacing", fontsize = 18)
plt.ylabel("Frequency", fontsize = 18)
fig.tight_layout()
plot_out = os.path.expanduser("~/GitHub/ParEvol") + '/figs/test_eigen_shape.png'
fig.savefig(plot_out, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
