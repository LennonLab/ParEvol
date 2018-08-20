from __future__ import division
import math, os, re
import numpy as np
import pandas as pd
import parevol_tools as pt
from collections import Counter
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table


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


def plot_nodes_over_time():
    directory = pt.get_path() + '/data/Good_et_al/networks_BIC/'
    time_nodes = []
    for filename in os.listdir(directory):
        df = pd.read_csv(directory + filename, sep = '\t', header = 'infer', index_col = 0)
        gens = filename.split('.')
        time = re.split('[_.]', filename)[1]
        #print(time)
        time_nodes.append((int(time), df.shape[0]))
    time_nodes_sorted = sorted(time_nodes, key=lambda tup: tup[0])
    x = [i[0] for i in time_nodes_sorted]
    y = [i[1] for i in time_nodes_sorted]

    fig = plt.figure()
    plt.scatter(x, y, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=3)
    #plt.plot(x, y)
    plt.xlabel("Time (generations)", fontsize = 16)
    plt.ylabel("Nodes", fontsize = 16)
    plt.yscale('log')
    #plt.ylim(0.001, 1)

    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/good_nodes_vs_time.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def plot_kmax_over_time():
    directory = pt.get_path() + '/data/Good_et_al/networks_BIC/'
    time_kmax = []
    for filename in os.listdir(directory):
        df = pd.read_csv(directory + filename, sep = '\t', header = 'infer', index_col = 0)
        gens = filename.split('.')
        time = re.split('[_.]', filename)[1]
        time_kmax.append((int(time), max(df.astype(bool).sum(axis=0).values)))

    time_kmax_sorted = sorted(time_kmax, key=lambda tup: tup[0])
    x = [i[0] for i in time_kmax_sorted]
    y = [i[1] for i in time_kmax_sorted]
    x = np.log10(x)
    y = np.log10(y)

    df_rndm_path = pt.get_path() + '/data/Good_et_al/networks_BIC_rndm.txt'
    df_rndm = pd.read_csv(df_rndm_path, sep = '\t', header = 'infer')

    x_rndm = np.log10(df_rndm.Generations.values)
    y_rndm = np.log10(df_rndm.Generations.values)

    fig = plt.figure()
    plt.scatter(x, y, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=3)
    plt.scatter(x_rndm, y_rndm, marker = "o", edgecolors='none', c = 'blue', s = 120, alpha = 0.1)

    '''using some code from ken locey, will cite later'''

    df = pd.DataFrame({'t': list(x)})
    df['kmax'] = list(y)
    f = smf.ols('kmax ~ t', df).fit()

    R2 = f.rsquared
    pval = f.pvalues
    intercept = f.params[0]
    slope = f.params[1]
    X = np.linspace(min(x), max(x), 1000)
    Y = f.predict(exog=dict(t=X))

    st, data, ss2 = summary_table(f, alpha=0.05)
    fittedvalues = data[:,2]
    pred_mean_se = data[:,3]
    pred_mean_ci_low, pred_mean_ci_upp = data[:,4:6].T
    pred_ci_low, pred_ci_upp = data[:,6:8].T


    plt.fill_between(x, pred_ci_low, pred_ci_upp, color='r', lw=0.5, alpha=0.2)
    plt.text(2, 0.2, r'$k_{max}$'+ ' = '+str(round(10**intercept,2))+'*'+r'$t$'+'$^{'+str(round(slope,2))+'}$', fontsize=10, color='Crimson', alpha=0.9)
    plt.text(2, 0.4,  r'$r^2$' + ' = ' +str("%.2f" % R2), fontsize=10, color='0.2')
    plt.plot(X.tolist(), Y.tolist(), '--', c='red', lw=2, alpha=0.8, color='Crimson', label='Power-law')

    #plt.plot(t_x, t_y)
    plt.xlabel("Time (generations)", fontsize = 16)
    plt.ylabel("k max", fontsize = 16)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim(0.001, 1)

    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/good_kmax_vs_time.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def get_kmax_random_networks(iterations = 1000):
    directory = pt.get_path() + '/data/Good_et_al/networks_BIC/'
    kmax_dict = {}
    df_out = open(pt.get_path() + '/data/Good_et_al/networks_BIC_rndm.txt', 'w')
    df_out.write('\t'.join(['Generations', 'Iteration', 'k_max']) + '\n')
    for filename in os.listdir(directory):
        df = pd.read_csv(directory + filename, sep = '\t', header = 'infer', index_col = 0)
        gens = filename.split('.')
        time = re.split('[_.]', filename)[1]
        iter_list = []
        print(time)
        for iter in range(iterations):
            random_matrix = pt.get_random_network(df)
            # -1 because the sum includes the node interacting with itself
            kmax_iter = int(max(np.sum(random_matrix, axis=0)) - 1)
            #iter_list.append(kmax_iter)
            df_out.write('\t'.join([str(time), str(iter), str(kmax_iter)]) + '\n')
        #kmax_dict[time] = iter_list
    df_out.close()

    #df = pd.DataFrame.from_dict(kmax_dict,orient='index').transpose()
    #df_path = pt.get_path() + '/data/Good_et_al/networks_BIC_rndm.txt'
    #df.to_csv(df_path, sep = '\t', index = True)


#plot_edge_dist()
#get_path_length()
#plot_nodes_over_time()
plot_kmax_over_time()
#get_kmax_random_networks()
