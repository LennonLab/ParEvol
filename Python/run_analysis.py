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


def permutation_analysis(iter, dataset = 'tenaillon'):
    if dataset == 'tenaillon':
        df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        df_array = df.as_matrix()
        #df_rndm = pt.random_matrix(df_array)
        iterations = []
        for i in range(iter):
            print(i)
            df_rndm = pd.DataFrame(data=pt.random_matrix(df_array), index=df.index, columns=df.columns)
            df_rndm_delta = pt.likelihood_matrix(df_rndm, 'Tenaillon_et_al').get_likelihood_matrix()

            df_rndm_delta_bc = np.sqrt(pt.get_bray_curtis(df_rndm_delta.as_matrix()))
            df_rndm_delta_cmd = pt.cmdscale(df_rndm_delta_bc)[0]

            iterations.append(pt.get_mean_centroid_distance(df_rndm_delta_cmd))

        df_delta = pt.likelihood_matrix(df, 'Tenaillon_et_al').get_likelihood_matrix()

        df_delta_bc = np.sqrt(pt.get_bray_curtis(df_delta.as_matrix()))
        df_delta_cmd = pt.cmdscale(df_delta_bc)[0]
        mcd = pt.get_mean_centroid_distance(df_delta_cmd)

        fig = plt.figure()
        plt.hist(iterations, bins=30, histtype='stepfilled', normed=True, alpha=0.6, color='b')
        plt.axvline(mcd, color = 'red')
        plt.xlabel("Mean centroid distance")
        plt.ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(pt.get_path() + '/figs/permutation_hist.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
        plt.close()

        #print(sum(df_array[1,:]))
        #print(sum(test[1,:]))
        #print(df[:,4])



#print(test_array)
#print(pt.random_matrix(test_array))
#permutation_analysis(1000)
