from __future__ import division
import math, os, re
import numpy as np
import pandas as pd
import parevol_tools as pt
import matplotlib.pyplot as plt
import clean_data as cd
from scipy import stats, spatial

from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics

mydir = os.path.expanduser("~/GitHub/ParEvol")

k_eval=2

def power_method_fig(alpha = 0.05):
    df = pd.read_csv(mydir + '/data/simulations/cov_ba_ntwrk_ev.txt', sep='\t')
    fig = plt.figure()
    covs = [0.05,0.1,0.15,0.2]
    measures = ['euc_percent', 'mcd_percent_k3', 'mcd_percent_k1', 'eig_percent']
    colors = ['midnightblue', 'blue',  'royalblue', 'skyblue']
    labels = ['Pairwise distance, ' + r'$k=1-3$',  'Centroid distance, ' + r'$k=1-3$', 'Centroid distance, ' + r'$k=1$', 'Largest eigenvalue']

    for i, measure in enumerate(measures):
        #df_i = df[ (df['Cov'] == cov) &  (df['Cov'] == cov)]
        powers = []
        for j, cov in enumerate(covs):
            df_cov = df[ df['Cov'] == cov ]
            p = df_cov[measure].values
            #p = df_i[ (df_i['N_genes_sample'] == gene_shuffle) ].p.tolist()
            p_sig = [p_i for p_i in p if p_i >= (1-  alpha)]
            powers.append(len(p_sig) / len(p))
        print(powers)
        plt.plot(np.asarray(covs), np.asarray(powers), linestyle='--', marker='o', color=colors[i], label=labels[i])
    #plt.title('Covariance', fontsize = 18)
    plt.legend(loc='upper left')
    plt.xlabel('Covariance between genes', fontsize = 16)
    plt.ylabel(r'$ \mathrm{P}\left ( \mathrm{reject} \; H_{0}   \mid H_{1} \;   \mathrm{is}\, \mathrm{true}, \, \alpha=0.05 \right ) $', fontsize = 16)
    #plt.xlim(-0.02, 1.02)
    #plt.ylim(-0.02, 1.02)
    plt.axhline(0.05, color = 'dimgrey', lw = 2, ls = '--')
    plt.tight_layout()
    fig_name = mydir + '/figs/power_method.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def tenaillon_PCA_fig(iter=10000):
    df_path = mydir + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    df_np_delta = cd.likelihood_matrix_array(df_np, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()
    X = df_np_delta/df_np_delta.sum(axis=1)[:,None]
    X -= np.mean(X, axis = 0)
    pca = PCA()
    df_out = pca.fit_transform(X)
    df_out_k = df_out[:,:k_eval]

    fig = plt.figure()
    # Scatterplot on main ax
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    ax1.axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)
    ax1.axvline(x=0, color='k', linestyle=':', alpha = 0.8, zorder=2)
    ax1.scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=3)
    ax1.scatter(df_out[:,0], df_out[:,1], marker = "o", edgecolors='#244162', c = '#175ac6', alpha = 0.4, s = 60, zorder=4)
    ax1.set_xlim([-0.7,0.7])
    ax1.set_ylim([-0.7,0.7])
    ax1.set_xlabel('PC 1 (' + str(round(pca.explained_variance_ratio_[0],3)*100) + '%)' , fontsize = 14)
    ax1.set_ylabel('PC 2 (' + str(round(pca.explained_variance_ratio_[1],3)*100) + '%)' , fontsize = 14)

    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    broken_stick = []
    for i in range(1, len(pca.explained_variance_ratio_) +1):
        broken_stick.append(   (sum(1 / np.arange(i, len(pca.explained_variance_) +1)) / len(pca.explained_variance_)) * 100   )
    broken_stick = np.asarray(broken_stick)
    broken_stick = broken_stick / sum(broken_stick)
    ax2.plot(list(range(1, len(pca.explained_variance_ratio_)+1)), pca.explained_variance_ratio_, linestyle='--', marker='o', color='royalblue', label='Observed')
    ax2.plot(list(range(1, len(pca.explained_variance_ratio_)+1)), broken_stick, linestyle=':', alpha=0.7, color='red', label='Broken-stick')
    ax2.legend(loc='upper right', fontsize='large')
    ax2.set_xlabel('Eigenvalue rank', fontsize = 14)
    ax2.set_ylabel('Proportion of\nvariance explained', fontsize = 12)

    # plot hist and get p value
    mpd_dist = pt.get_mean_pairwise_euc_distance(df_out)
    mpd_null_dists = []
    ks = range(2, 11)
    ch_scores = []
    ch_null_scores = []
    for k in ks:
        kmeans_model = KMeans(n_clusters=k).fit(df_out_k)
        ch_scores.append(metrics.calinski_harabasz_score(X, kmeans_model.labels_))
    for i in range(iter):
        if i%100 == 0:
            print(i)
        df_np_i = pt.get_random_matrix(df_np)
        df_np_delta_i = cd.likelihood_matrix_array(df_np_i, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()
        X_i = df_np_delta_i/df_np_delta_i.sum(axis=1)[:,None]
        X_i -= np.mean(X_i, axis = 0)
        df_out_i = pca.fit_transform(X_i)
        np.seterr(divide='ignore')
        mpd_null_dists.append( pt.get_mean_pairwise_euc_distance(df_out_i) )
        df_out_i_k = df_out_i[:,:k_eval]
        ch_scores_null_i = []
        for k in ks:
            # Create and fit a KMeans instance with k clusters: model
            kmeans_model_null = KMeans(n_clusters=k).fit(df_out_i_k)
            # get Calinski-Harabasz Index
            ch_scores_null_i.append(metrics.calinski_harabasz_score(X, kmeans_model_null.labels_))
        ch_null_scores.append(ch_scores_null_i)

    mpd_greater = [j for j in mpd_null_dists if j > mpd_dist]
    p_value = len(mpd_greater) / iter
    print(p_value)

    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax3.hist(mpd_null_dists, bins=30, weights=np.zeros_like(mpd_null_dists) + 1. / len(mpd_null_dists), alpha=0.8, color = '#175ac6')
    ax3.axvline(mpd_dist, color = 'red', lw = 3, ls = '--')
    ax3.set_xlabel("Mean pairwise distance", fontsize = 14)
    ax3.set_ylabel("Frequency", fontsize = 16)
    #ax3.text(0.22, 0.08, r'$p = $' + str(round(p_value, 4)), fontsize = 8)

    ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    for ch_null_score in ch_null_scores:
        ax4.plot(list(range(2, 11)), ch_null_score, linestyle='-', alpha=0.4, color='royalblue')
    ax4.plot(list(range(2, 11)), ch_scores, linestyle='--', marker='o', color='red')
    ax4.set_xlabel('Number of clusters', fontsize = 12)
    ax4.set_ylabel('Variance ratio criterion', fontsize = 12)

    plt.tight_layout()
    fig_name = mydir + '/figs/fig1.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def tenaillon_fitnes_fig():
    df_path = mydir + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    df_np_delta = cd.likelihood_matrix_array(df_np, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()
    X = df_np_delta/df_np_delta.sum(axis=1)[:,None]
    X -= np.mean(X, axis = 0)
    pca = PCA()
    df_out = pca.fit_transform(X)
    PC_1_2 = df_out[:,:2]
    PC_1_2_df = pd.DataFrame({'PC1':df_out[:,0], 'PC2':df_out[:,1]}, index = df.index)
    fitness_path =mydir + '/data/Tenaillon_et_al/fitness.csv'
    fitness = pd.read_csv(fitness_path, sep = ',', header = 'infer', index_col = 0)
    PC_1_2_W_df = PC_1_2_df.merge(fitness, left_index=True, right_index=True)
    print(PC_1_2_W_df)

    PC1 = np.asarray(PC_1_2_W_df.PC1.values)
    PC2 = np.asarray(PC_1_2_W_df.PC2.values)
    W = np.asarray(PC_1_2_W_df['W (avg)'].values)

    slope_PC1_W, intercept_PC1_W, r_value_PC1_W, p_value_PC1_W, std_err_PC1_W = stats.linregress(PC1,W)
    slope_PC2_W, intercept_PC2_W, r_value_PC2_W, p_value_PC2_W, std_err_PC2_W = stats.linregress(PC2,W)

    print("PC1 vs. W slope is " + str(round(slope_PC1_W, 3)))
    print("PC1 vs. W r^2 is " + str(round(r_value_PC1_W **2, 3)))
    print("PC1 vs. W p-value is " + str(round(p_value_PC1_W, 3)))

    print("PC2 vs. W slope is " + str(round(slope_PC2_W, 3)))
    print("PC2 vs. W r^2 is " + str(round(r_value_PC2_W **2, 3)))
    print("PC2 vs. W p-value is " + str(round(p_value_PC2_W, 3)))

    fig = plt.figure()
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    #ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1)
    axes[0].scatter(PC1, W, c='#175ac6', marker = 'o', s = 70, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)#, edgecolors='none')
    axes[0].set_xlabel("PC 1", fontsize = 18)
    axes[0].set_ylabel("Fitness, " + r'$W$', fontsize = 18)
    axes[0].text(0.37, 1.75, r'$p \nless 0.05$', fontsize = 10)
    axes[0].set(adjustable='box-forced', aspect='equal')

    axes[1].scatter(PC2, W, c='#175ac6', marker = 'o', s = 70, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)#, edgecolors='none')
    axes[1].set_xlabel("PC 2", fontsize = 18)
    axes[1].set_ylabel("Fitness, " + r'$W$', fontsize = 18)
    axes[1].text(0.37, 1.75, r'$p \nless 0.05$', fontsize = 10)
    axes[1].set(adjustable='box-forced', aspect='equal')

    fig.tight_layout()
    plot_path = mydir + '/figs/tenaillon_PC_W.png'
    fig.savefig(plot_path, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def tenaillon_stability_fig(iter=1000):
    df_path = mydir + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    df_np_delta = cd.likelihood_matrix_array(df_np, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()
    X = df_np_delta/df_np_delta.sum(axis=1)[:,None]
    X -= np.mean(X, axis = 0)
    pca = PCA()
    df_out = pca.fit_transform(X)
    df_out_k = df_out[:,:k_eval]

    df_out_k_kmeans = SpectralClustering(n_clusters=3).fit(df_out_k)
    k_dict = {i: np.where(df_out_k_kmeans.labels_ == i)[0].tolist() for i in range(df_out_k_kmeans.n_clusters)}

    jaccard_dict = {'0':[], '1':[], '2':[]}
    for i in range(iter):
        df_out_k_i = df_out_k[np.random.choice(df_out_k.shape[0], df_out_k.shape[0])]
        df_out_k_i_kmeans = SpectralClustering(n_clusters=3).fit(df_out_k_i)
        k_i_dict = {i: np.where(df_out_k_i_kmeans.labels_ == i)[0].tolist() for i in range(df_out_k_i_kmeans.n_clusters)}
        for key, value in k_dict.items():
            jaccards_key = []
            for key_i, value_i in k_i_dict.items():
                jaccards_key.append((pt.jaccard_similarity(value, value_i)))
            jaccard_dict[str(key)].append(max(jaccards_key))

    fig = plt.figure()
    plt.boxplot([v for k,v in jaccard_dict.items()])
    for i in [1,2,3]:
        y = jaccard_dict[str(i-1)]
        x = np.random.normal(i, 0.04, len(y)).tolist()
        plt.scatter(x, y, c='b', alpha = 0.1)
    plt.ylim([0,1])
    plt.xlabel("Cluster", fontsize = 18)
    plt.ylabel("Jaccard similarity index", fontsize = 16)

    fig.tight_layout()
    plot_path = mydir + '/figs/tenaillon_stability.png'
    fig.savefig(plot_path, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def plot_ltee_partition(k = 5):
    # mpd
    df_path = mydir + '/data/Good_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    to_keep = pt.complete_nonmutator_lines()
    #if 'p5' in to_keep:
    #    to_keep.remove('p5')
    df_nonmut = df[df.index.str.contains('|'.join( to_keep))]
    # remove columns with all zeros
    df_nonmut = df_nonmut.loc[:, (df_nonmut != 0).any(axis=0)]
    gene_names = df_nonmut.columns.tolist()
    sample_names = df_nonmut.index.tolist()
    df_delta = cd.likelihood_matrix_array(df_nonmut, gene_names, 'Good_et_al').get_likelihood_matrix()
    X = df_delta/df_delta.sum(axis=1)[:,None]
    X -= np.mean(X, axis = 0)
    pca = PCA()
    df_out = pca.fit_transform(X)
    time_points = [ int(x.split('_')[1]) for x in df_nonmut.index.values]
    time_points_set = sorted(list(set([ int(x.split('_')[1]) for x in df_nonmut.index.values])))
    colors = np.linspace(min(time_points_set),max(time_points_set),len(time_points_set))
    color_dict = dict(zip(time_points_set, colors))
    df_pca = pd.DataFrame(data=df_out, index=sample_names)
    mean_dist = []
    for tp in time_points_set:
        df_pca_tp = df_pca[df_pca.index.str.contains('_' + str(tp))]
        mean_dist.append(pt.get_mean_pairwise_euc_distance(df_pca_tp.values, k = k))

    df_path = mydir + '/data/Good_et_al/time_partition_z_scores.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer')

    time = df.Time.values
    t_mpd = df.delta_mpd.values
    t_less_025 = df.delta_mpd_025.values
    t_less_975 = df.delta_mpd_975.values
    t_greater = df.greater_mpd.values

    fig, axes = plt.subplots(2, sharex=True)
    axes[0].scatter(time_points_set, mean_dist, marker = "o", edgecolors='#244162', c = '#175ac6', alpha = 0.4, s = 60, zorder=4)
    axes[0].set_ylabel("Mean pairwise\ndistance, " + r'$d$', fontsize = 14)
    axes[0].axvline(x=10000, c='dimgrey', ls='--')

    axes[1].plot(time, t_mpd, c ='k')
    axes[1].fill_between(time, t_less_025, t_less_975, facecolor='blue', alpha=0.5)
    axes[1].axvline(x=10000, c='dimgrey', ls='--')
    axes[1].set_xlabel("Time (generations)", fontsize = 18)
    axes[1].set_ylabel(r'$\Delta d$', fontsize = 18)
    axes[1].set_xlim( right=51000)


    fig.tight_layout()
    fig.savefig(mydir + '/figs/ltee_partition.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def tenaillon_gene_pca():
    df_genes_path = mydir + '/data/Tenaillon_et_al/gene_z_scores.txt'
    df_genes = pd.read_csv(df_genes_path, sep = '\t', header = 'infer', index_col = 0)
    df_genes = df_genes.loc[df_genes['p_score'] < 0.05]
    print(df_genes)
    df_path = mydir + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    df_np_delta = cd.likelihood_matrix_array(df_np, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()
    X = df_np_delta/df_np_delta.sum(axis=1)[:,None]
    X -= np.mean(X, axis = 0)
    pca = PCA()
    df_out = pca.fit_transform(X)
    coords = list(zip(df.index.tolist(), df_out[:,0].tolist(), df_out[:,1].tolist()))

    cluster_1 = [x[0] for x in coords if x[1] < 0.2 and x[2] < 0.08]
    cluster_1_df = df.loc[cluster_1]
    cluster_1_df = cluster_1_df.loc[:, (cluster_1_df != 0).any(axis=0)]
    #rpoB

    cluster_2 = [x[0] for x in coords if x[1] > 0.2]
    cluster_2_df = df.loc[cluster_2]
    cluster_2_df = cluster_2_df.loc[:, (cluster_2_df != 0).any(axis=0)]
    #print(cluster_2_df.sum(axis=0).sort_values())
    # cluster 2 ESCRE1901
    cluster_3 = [x[0] for x in coords if x[1] < 0.2 and x[2] > 0.08]
    cluster_3_df = df.loc[cluster_3]
    cluster_3_df = cluster_3_df.loc[:, (cluster_3_df != 0).any(axis=0)]
    # cluster 3 ECB_01992

    df_pops_rpoB = df.loc[df['rpoB'] > 0].index.tolist()
    df_pops_ybaL = df.loc[df['ybaL'] > 0].index.tolist()
    df_pops_ESCRE1901 = df.loc[df['ESCRE1901'] > 0].index.tolist()
    df_pops_ECB_01992 = df.loc[df['ECB_01992'] > 0].index.tolist()
    colors_rpoB = ['r' if i in df_pops_rpoB else '#175ac6' for i in df.index.tolist()]
    colors_ybaL = ['r' if i in df_pops_ybaL else '#175ac6' for i in df.index.tolist()]
    colors_ESCRE1901 = ['r' if i in df_pops_ESCRE1901 else '#175ac6' for i in df.index.tolist()]
    colors_ECB_01992 = ['r' if i in df_pops_ECB_01992 else '#175ac6' for i in df.index.tolist()]

    fig, axes = plt.subplots(2, 2)

    axes[0][0].axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)
    axes[0][0].axvline(x=0, color='k', linestyle=':', alpha = 0.8, zorder=2)
    axes[0][0].scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', s = 80, zorder=3)
    axes[0][0].scatter(df_out[:,0], df_out[:,1], marker = "o", edgecolors='#244162', c = colors_rpoB, alpha = 0.4, s = 60, zorder=4)
    axes[0][0].text(0.28,0.4,r'$rpoB$')
    axes[0][0].text(0.28,0.32,r'$n_{pop}=$' + str(len(df_pops_rpoB)))

    axes[0][1].axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)
    axes[0][1].axvline(x=0, color='k', linestyle=':', alpha = 0.8, zorder=2)
    axes[0][1].scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', s = 80, zorder=3)
    axes[0][1].scatter(df_out[:,0], df_out[:,1], marker = "o", edgecolors='#244162', c = colors_ybaL, alpha = 0.4, s = 60, zorder=4)
    axes[0][1].text(0.28,0.4,r'$ybaL$')
    axes[0][1].text(0.28,0.32,r'$n_{pop}=$' + str(len(df_pops_ybaL)))

    axes[1][0].axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)
    axes[1][0].axvline(x=0, color='k', linestyle=':', alpha = 0.8, zorder=2)
    axes[1][0].scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', s = 80, zorder=3)
    axes[1][0].scatter(df_out[:,0], df_out[:,1], marker = "o", edgecolors='#244162', c = colors_ESCRE1901, alpha = 0.4, s = 60, zorder=4)
    axes[1][0].text(0.28,0.4,r'$\mathrm{ESCRE1901}$')
    axes[1][0].text(0.28,0.32,r'$n_{pop}=$' + str(len(df_pops_ESCRE1901)))

    axes[1][1].axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)
    axes[1][1].axvline(x=0, color='k', linestyle=':', alpha = 0.8, zorder=2)
    axes[1][1].scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', s = 80, zorder=3)
    axes[1][1].scatter(df_out[:,0], df_out[:,1], marker = "o", edgecolors='#244162', c = colors_ECB_01992, alpha = 0.4, s = 60, zorder=4)
    axes[1][1].text(0.28,0.4,r'$\mathrm{ECB01992}$')
    axes[1][1].text(0.28,0.32,r'$n_{pop}=$' + str(len(df_pops_ECB_01992)))


    fig.text(0.5, 0.0001, 'PC 1 (' + str(round(pca.explained_variance_ratio_[0],3)*100) + '%)', ha='center', fontsize=18)
    fig.text(0.0001, 0.5, 'PC 2 (' + str(round(pca.explained_variance_ratio_[1],3)*100) + '%)', va='center', fontsize=18, rotation='vertical')


    #fig.tight_layout()
    fig.savefig(mydir + '/figs/tenaillon_gene_pca.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()

    #print(df_pops_mrdA.index.tolist())





tenaillon_gene_pca()
