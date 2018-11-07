from __future__ import division
import math, random, itertools
import numpy as np
import pandas as pd
import scipy.stats as stats
import parevol_tools as pt
import matplotlib.pyplot as plt


df_path = pt.get_path() + '/data/simulate_2envs.txt'
df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)

genes_shuffle = list( range(2, 50, 5)  )
muts = [10, 50, 100, 150, 200]
pops = [5, 10, 15, 20, 25]

colors = ['powderblue', 'skyblue', 'royalblue', 'blue', 'navy']


def plot_lines(mut):
    fig = plt.figure()
    for i, pop in enumerate(pops):
        df_i = df[ (df['N_muts'] == mut) & (df['N_pops'] == pop)  ]
        powers = []
        for gene_shuffle in genes_shuffle:
            p = df_i[ (df_i['N_genes_sample'] == gene_shuffle) ].p.tolist()
            p_sig = [p_i for p_i in p if p_i < 0.05]
            powers.append(len(p_sig) / len(p))
        p_genes_shuffle = [i / 50 for i in genes_shuffle]
        plt.plot(p_genes_shuffle, powers, linestyle='--', marker='o', color=colors[i], label='N = ' + str(pop))
    plt.title('Substitutions = ' + str(mut), fontsize = 18)
    plt.legend(loc='lower right')
    plt.xlabel('Proportion of substitution rates shuffled', fontsize = 16)
    plt.ylabel(r'$ \mathrm{Pr}\left ( \mathrm{reject} \; H_{0}   \mid H_{1} \;   \mathrm{is}\, \mathrm{true} \right ) $', fontsize = 16)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    fig_name = pt.get_path() + '/figs/rda_m' + str(mut) + '.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()





plot_lines(10)
plot_lines(50)



#print(df_M10_P5)

#df_M10_P5.groupby(['N_genes_sample'])['p'].mean().reset_index()
