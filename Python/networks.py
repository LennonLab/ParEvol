from __future__ import division
import math
import numpy as np
import pandas as pd
import parevol_tools as pt
from collections import Counter
import matplotlib.pyplot as plt

def plot_edge_dist():
    df_path = pt.get_path() + '/data/Tenaillon_et_al/network.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    k_list = []
    for index, row in df.iterrows():
        k_row = sum(i >0 for i in row.values) - 1
        if k_row > 0:
            k_list.append(k_row)

    k_count = dict(Counter(k_list))
    k_count = {k: v / total for total in (sum(k_count.values()),) for k, v in k_count.items()}
    #x = np.log10(list(k_count.keys()))
    #y = np.log10(list(k_count.values()))
    k_mean = np.mean(k_list)
    print("mean k = " + str(k_mean))
    print("N = " + str(df.shape[0]))
    x = list(k_count.keys())
    y = list(k_count.values())

    x_poisson = list(range(1, 100))
    y_poisson = [(math.exp(-k_mean) * ( (k_mean ** k)  /  math.factorial(k) )) for k in x_poisson]

    fig = plt.figure()
    plt.scatter(x, y, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=3)
    plt.plot(x_poisson, y_poisson)
    plt.xlabel("Number of edges, k", fontsize = 16)
    plt.ylabel("Frequency", fontsize = 16)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(0.001, 1)

    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/edge_dist.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def get_path_length():
    df_path = pt.get_path() + '/data/Tenaillon_et_al/network.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    k_list = []
    #print(df)
    node_dict = {}
    #df['puuD'].values
    count_i = 0
    count_j = 0
    for index_i, row_i in df.iterrows():
        count_i += 1
        count_j += 1

        for index_j, row_j in df.iterrows():
            continue
    print(count_i, count_j)
    #        if index_i == index_j:
    #            print(index_i, index_j)
    #    k_row = sum(i >0 for i in row.values) - 1
    #    if k_row > 0:
    #        k_list.append(k_row)
    #print(np.mean(k_list))


plot_edge_dist()
#get_path_length()
