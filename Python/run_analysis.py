from __future__ import division
import numpy as np
import pandas as pd
import parevol_tools as pt
import matplotlib.pyplot as plt



test_array = np.array([[1, 2], [3, 4], [3, 0]])


def plot_pcoa():
    df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop_delta.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    # square root transformation to remove negative eigenvalues
    X = np.sqrt(pt.get_bray_curtis(df.as_matrix()))
    X_cmds = pt.cmdscale(X)
    #pt.plot_eigenvalues(X_cmds[1], file_name = 'pcoa_tenaillon_eigen')
    percent_explained = X_cmds[1] / sum(X_cmds[1])

    names = df.index.values

    fig = plt.figure()
    plt.axhline(y=0, color='k', linestyle=':', alpha = 0.8)
    plt.axvline(x=0, color='k', linestyle=':', alpha = 0.8)
    plt.scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', alpha = 0.8, s = 100)
    plt.scatter(X_cmds[0][:,0], X_cmds[0][:,1], marker = "o", edgecolors='none', c = 'forestgreen', alpha = 0.6, s = 80)


    plt.xlim([-0.4,0.4])
    plt.ylim([-0.4,0.4])
    plt.xlabel('PCoA 1 (' + str(round(percent_explained[0],3)*100) + '%)' , fontsize = 16)
    plt.ylabel('PCoA 2 (' + str(round(percent_explained[1],3)*100) + '%)' , fontsize = 16)
    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/test_pcoa.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def permutation_analysis(dataset = 'tenaillon'):
    if dataset == 'tenaillon':
        df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        df_array = df.as_matrix()
        #print( sum(df_array.sum(axis=1)))
        #print( sum(df_array.sum(axis=0)))
        #print(sum(df_array[0,:]))
        #df_rndm = pt.sis_matrix(df_array)

        #print(sum(df_rndm[0,:]))
        #print(pd.DataFrame(data=df_rndm, index=df.index, columns=df.columns).as_matrix()[1,:])
        #df_rndm_delta = pt.likelihood_matrix(df_rndm).get_likelihood_matrix()

        #print(sum(df_array[1,:]))
        #print(sum(test[1,:]))
        #print(df[:,4])
print(test_array)
print(pt.random_matrix(test_array))

#permutation_analysis()
