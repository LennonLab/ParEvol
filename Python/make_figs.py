from __future__ import division
import math, os, re, functools
import numpy as np
import pandas as pd
import parevol_tools as pt
import matplotlib.pyplot as plt
from matplotlib import cm
#from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import clean_data as cd
from scipy import stats

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics

np.random.seed(123456789)

mydir = os.path.expanduser("~/GitHub/ParEvol")

k_eval=3


def tenaillon_sig_multiplicity_fig():
    df = pd.read_csv(mydir + '/data/Tenaillon_et_al/sig_genes_sim.txt', sep='\t', header = 'infer')
    fig = plt.figure(figsize = (5, 4))
    fig.tight_layout(pad = 2.8)

    # G score fig
    # G-score for 115 pops = 1.0974618847690791
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    ax1.errorbar(df.N.values, df.n_mut_mean.values, yerr = [df.n_mut_mean.values-df.n_mut_ci_975.values, df.n_mut_ci_025.values - df.n_mut_mean.values ], \
        fmt = 'o', alpha = 1, \
        barsabove = True, marker = '.',  ls = "None", mfc = 'k', mec = 'k', c = 'k', zorder=2)
    ax1.scatter(df.N.values, df.n_mut_mean.values, c='#175ac6', marker = 'o', s = 2, \
        edgecolors='none', linewidth = 0.6, alpha = 1, zorder=3)
    #ax1.set_xlabel('Number of replicate populations', fontsize = 10)
    ax1.set_ylabel('Net increase in\nlog-likelihood, ' + r'$\Delta \ell$', fontsize = 10)
    ax1.axhline(1.0974618847690791, color = 'red', lw = 2, ls = '--', zorder=1, label = 'All populations')
    ax1.axhline(0, color = 'dimgrey', lw = 2, ls = ':', zorder=1, label = 'Null')

    ax1.legend(loc='upper right', fontsize=7)


    ax1.set_ylim(-0.2,3)

    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    ax2.errorbar(df.N.values, df.genes_mean.values, yerr = [df.genes_mean.values-df.genes_mean_ci_975.values, df.genes_mean_ci_025.values - df.genes_mean.values ], \
        fmt = 'o', alpha = 1, \
        barsabove = True, marker = '.',  ls = "None", mfc = 'k', mec = 'k', c = 'k', zorder=2)
    ax2.scatter(df.N.values, df.genes_mean.values, c='#175ac6', marker = 'o', s = 2, \
        edgecolors='none', linewidth = 0.6, alpha = 1, zorder=3)
    ax2.set_ylabel('Number of genes with\nsignificant multiplicity', fontsize = 8)
    ax2.axhline(28, color = 'red', lw = 2, ls = '--', zorder=1, label = 'All populations')
    ax2.legend(loc='lower right', fontsize=7)
    ax2.set_ylim(-2,32)

    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax3.errorbar(df.N.values, df.ESCRE1901_mean.values, yerr = [df.ESCRE1901_mean.values-df.ESCRE1901_ci_975.values, df.ESCRE1901_ci_025.values - df.ESCRE1901_mean.values ], \
        fmt = 'o', alpha = 1, \
        barsabove = True, marker = '.',  ls = "None", mfc = 'k', mec = 'k', c = 'k', zorder=2)
    ax3.scatter(df.N.values, df.ESCRE1901_mean.values, c='#175ac6', marker = 'o', s = 2, \
        edgecolors='none', linewidth = 0.6, alpha = 1, zorder=3)
    ax3.set_ylabel('Proportion of times ESCRE1901\nhad significant multiplicity', fontsize = 8)
    ax3.set_ylim(-0.05,1.05)

    ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax4.errorbar(df.N.values, df.ECB_01992_mean.values, yerr = [df.ECB_01992_mean.values-df.ECB_01992_ci_975.values, df.ECB_01992_ci_025.values - df.ECB_01992_mean.values ], \
        fmt = 'o', alpha = 1, \
        barsabove = True, marker = '.',  ls = "None", mfc = 'k', mec = 'k', c = 'k', zorder=2)
    ax4.scatter(df.N.values, df.ECB_01992_mean.values, c='#175ac6', marker = 'o', s = 2.5, \
        edgecolors='none', linewidth = 0.1, alpha = 1, zorder=3)
    ax4.set_ylabel('Proportion of times ECB_01992\nhad significant multiplicity', fontsize = 8)
    ax4.set_ylim(-0.05,1.05)


    fig.text(0.5, -0.04,'Number of replicate populations', fontsize = 16, ha='center')

    ax1.text(-0.1, 1.1, "a", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax1.transAxes)
    ax2.text(-0.1, 1.1, "b", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax2.transAxes)
    ax3.text(-0.1, 1.1, "c", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax3.transAxes)
    ax4.text(-0.1, 1.1, "d", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax4.transAxes)


    plt.tight_layout()
    fig.savefig(mydir + '/figs/tenaillon_sig_multiplicity.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def power_method_fig(alpha = 0.05):
    df = pd.read_csv(mydir + '/data/simulations/cov_ba_ntwrk_methods.txt', sep='\t')
    fig = plt.figure(figsize = (6, 6))
    fig.tight_layout(pad = 2.8)

    # Scatterplot on main ax
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    #fig = plt.figure()
    measures = ['Eig', 'MCD_k1', 'MCD_k3', 'MPD_k1', 'MPD_k3']
    colors = ['k', 'firebrick', 'orangered',  'dodgerblue', 'lightskyblue']
    labels = [r'$\tilde{L}_{1}$', r'$\mathrm{MCD}^{\left ( 1 \right )}$', r'$\mathrm{MCD}^{\left ( 3 \right )}$', r'$\mathrm{MPD}^{\left ( 1 \right )}$', r'$\mathrm{MPD}^{\left ( 3 \right )}$' ]
    for i, measure in enumerate(measures):
        df_i = df[ df['Method'] == measure ]
        color_i = colors[i]
        cov_jitter = df_i.Cov.values + np.random.normal(0, 0.003, len(df_i.Cov.values))
        ax1.errorbar(cov_jitter, df_i.Power.values, yerr = [df_i.Power.values-df_i.Power_975.values, df_i.Power_025.values - df_i.Power.values ], \
            fmt = 'o', alpha = 1, ms=20,\
            marker = '.', mfc = color_i,
            mec = color_i, c = 'k', zorder=2, label=labels[i])

    ax1.legend(loc='upper left')
    ax1.set_xlabel('Covariance', fontsize = 10)
    ax1.set_ylabel("Statistical power\n" +r'$ \mathrm{P}\left ( \mathrm{reject} \; H_{0}   \mid H_{1} \;   \mathrm{is}\, \mathrm{true}, \, \alpha=0.05 \right ) $', fontsize = 10)
    ax1.set_xlim(0.02, 0.22)
    ax1.set_ylim(-0.02, 0.48)
    ax1.axhline(0.05, color = 'dimgrey', lw = 2, ls = '--', zorder=1)

    # figure 2
    df_cluster = pd.read_csv(mydir + '/data/simulations/cov_ba_ntwrk_cluster_methods.txt', sep='\t')
    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    for i, measure in enumerate(measures):
        df_cluster_i = df_cluster[ df_cluster['Method'] == measure ]
        color_i = colors[i]
        x_jitter = df_cluster_i.CC_mean.values + np.random.normal(0, 0.003, len(df_cluster_i.CC_mean.values))
        ax2.errorbar(x_jitter, df_cluster_i.Power.values, yerr = [df_cluster_i.Power.values-df_cluster_i.Power_975.values, df_cluster_i.Power_025.values - df_cluster_i.Power.values ], \
            fmt = 'o', alpha = 1, ms=20,\
            marker = '.', mfc = color_i,
            mec = color_i, c = 'k', zorder=2, label=labels[i])


    ax2.set_xlabel('Mean clustering coefficient', fontsize = 10)
    ax2.axhline(0.05, color = 'dimgrey', lw = 2, ls = '--', zorder=1)
    ax2.set_ylim(-0.02, 0.48)
    ax2.set_xlim(0.12, 0.82)


    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    for i, measure in enumerate(measures):
        df_i = df[ df['Method'] == measure ]
        color_i = colors[i]
        cov_jitter = df_i.Cov.values + np.random.normal(0, 0.003, len(df_i.Cov.values))
        ax3.errorbar(cov_jitter, df_i.Z_mean.values, yerr = [df_i.Z_mean.values-df_i.Z_975.values, df_i.Z_025.values - df_i.Z_mean.values ], \
            fmt = 'o', alpha = 1, ms=20,\
            marker = '.', mfc = color_i,
            mec = color_i, c = 'k', zorder=2, label=labels[i])

    ax3.axhline(0, color = 'dimgrey', lw = 2, ls = '--', zorder=1)
    ax3.axhline(1, color = 'black', lw = 2, ls = ':', zorder=1)
    ax3.set_xlim(0.02, 0.22)
    ax3.set_ylim(-0.38, 1.38)
    ax3.set_xlabel('Covariance', fontsize = 10)
    ax3.set_ylabel("Standard score", fontsize = 10)



    ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1)

    for i, measure in enumerate(measures):
        df_cluster_i = df_cluster[ df_cluster['Method'] == measure ]
        color_i = colors[i]
        x_jitter = df_cluster_i.CC_mean.values + np.random.normal(0, 0.003, len(df_cluster_i.CC_mean.values))
        ax4.errorbar(x_jitter, df_cluster_i.Z_mean.values, yerr = [df_cluster_i.Z_mean.values-df_cluster_i.Z_975.values, df_cluster_i.Z_025.values - df_cluster_i.Z_mean.values ], \
            fmt = 'o', alpha = 1, ms=20,\
            marker = '.', mfc = color_i,
            mec = color_i, c = 'k', zorder=2, label=labels[i])

    ax4.axhline(0, color = 'dimgrey', lw = 2, ls = '--', zorder=1)
    ax4.axhline(1, color = 'black', lw = 2, ls = ':', zorder=1)
    ax4.set_xlim(0.12, 0.82)
    ax4.set_ylim(0.25,1.5)
    ax4.set_xlabel('Mean clustering coefficient', fontsize = 10)

    #ax4.set_ylabel("Standard score", fontsize = 10)

    ax1.text(-0.1, 1.07, "a", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax1.transAxes)
    ax2.text(-0.1, 1.07, "b", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax2.transAxes)
    ax3.text(-0.1, 1.07, "c", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax3.transAxes)
    ax4.text(-0.1, 1.07, "d", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax4.transAxes)


    plt.tight_layout()
    fig.savefig(mydir + '/figs/power_method.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def tenaillon_corr_PCA_fig(iter=10000):
    df_path = mydir + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    df_np_delta = cd.likelihood_matrix_array(df_np, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()

    X = df_np_delta/df_np_delta.sum(axis=1)[:,None]
    X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
    pca = PCA()
    df_out = pca.fit_transform(X)
    df_out_k = df_out[:,:k_eval]

    eigen = pt.get_x_stat(pca.explained_variance_[:-1], n_features=len(gene_names))

    # get null data
    mpd_dist = pt.get_mean_pairwise_euc_distance(df_out)
    mpd_null_dists = []
    #ks = range(2, 11)
    #ch_scores = []
    #ch_null_scores = []
    #variance_ratios_null = []
    eigen_null = []
    #for k in ks:
    #    kmeans_model = KMeans(n_clusters=k).fit(df_out_k)
    #    ch_scores.append(metrics.calinski_harabasz_score(X, kmeans_model.labels_))
    for i in range(iter):
        if i%100 == 0:
            print(i)
        df_np_i = pt.get_random_matrix(df_np)
        df_np_delta_i = cd.likelihood_matrix_array(df_np_i, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()
        X_i = df_np_delta_i/df_np_delta_i.sum(axis=1)[:,None]
        X_i = (X_i - np.mean(X_i, axis = 0)) / np.std(X_i, axis = 0)
        pca_i = PCA()
        df_out_i = pca_i.fit_transform(X_i)
        np.seterr(divide='ignore')
        mpd_null_dists.append( pt.get_mean_pairwise_euc_distance(df_out_i) )

        eigen_null.append(pt.get_x_stat(pca_i.explained_variance_[:-1], n_features=len(gene_names)))

        #df_out_i_k = df_out_i[:,:k_eval]
        #ch_scores_null_i = []
        #for k in ks:
        #    # Create and fit a KMeans instance with k clusters: model
        #    kmeans_model_null = KMeans(n_clusters=k).fit(df_out_i_k)
        #    # get Calinski-Harabasz Index
        #    ch_scores_null_i.append(metrics.calinski_harabasz_score(X, kmeans_model_null.labels_))
        #ch_null_scores.append(ch_scores_null_i)


    mpd_greater = [j for j in mpd_null_dists if j > mpd_dist]
    p_value = (len(mpd_greater) +1) / (iter+1)
    z_score = (mpd_dist - np.mean(mpd_null_dists)) / np.std(mpd_null_dists)
    print("p-value = " +  str(p_value))
    print("z-value = " +  str(z_score))

    eigen_greater = [j for j in eigen_null if j > eigen]
    p_value_eigen = (len(eigen_greater) +1) / (iter+1)
    z_score_eigen = (eigen - np.mean(eigen_null)) / np.std(eigen_null)


    fig = plt.figure(figsize = (6, 6))
    fig.tight_layout(pad = 2.8)
    # Scatterplot on main ax
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    ax1.axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)
    ax1.axvline(x=0, color='k', linestyle=':', alpha = 0.8, zorder=2)

    df_pops_ESCRE1901 = df.loc[df['ESCRE1901'] > 0].index.tolist()
    df_pops_ECB_01992 = df.loc[df['ECB_01992'] > 0].index.tolist()

    df_out_labelled = pd.DataFrame(data=df_out, index=df.index.values)
    df_pops_ESCRE1901_pc = df_out_labelled.loc[ df_pops_ESCRE1901 , : ]
    df_pops_ECB_01992_pc = df_out_labelled.loc[ df_pops_ECB_01992 , : ]

    df_out_labelled_no_clsust = df_out_labelled.drop(df_pops_ESCRE1901+df_pops_ECB_01992)


    ax1.scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=3)
    ax1.set_xlim([-20, 80 ])
    ax1.set_ylim([-20, 50 ])

    ax1.set_xlabel('PC 1 (' + str(round(pca.explained_variance_ratio_[0]*100,2)) + '%)' , fontsize = 12)
    ax1.set_ylabel('PC 2 (' + str(round(pca.explained_variance_ratio_[1]*100,2)) + '%)' , fontsize = 12)

    ax1.scatter(df_out_labelled_no_clsust.values[:,0], df_out_labelled_no_clsust.values[:,1], marker = "o", edgecolors='#244162',  c = '#175ac6', alpha = 0.3, s = 60, zorder=4)

    ax1.scatter(df_pops_ESCRE1901_pc.values[:,0], df_pops_ESCRE1901_pc.values[:,1], marker = "o", edgecolors='#244162',  c = 'r', alpha = 0.6, s = 60, zorder=5)
    ax1.scatter(df_pops_ECB_01992_pc.values[:,0], df_pops_ECB_01992_pc.values[:,1], marker = "o", edgecolors='#244162',  c = 'purple', alpha = 0.6, s = 60, zorder=5)


    ax1.text(0.7,0.8,r'$n_{\mathrm{ESCRE1901}}=$' + str(len(df_pops_ESCRE1901)), fontsize=11, color='r', ha='center', va='center', transform=ax1.transAxes  )
    ax1.text(0.7,0.7,r'$n_{\mathrm{ECB \_ 01992 }} = $' + str(len(df_pops_ECB_01992)), fontsize=11, color='purple', ha='center', va='center', transform=ax1.transAxes)

    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    ax2.hist(mpd_null_dists, bins=30, weights=np.zeros_like(mpd_null_dists) + 1. / len(mpd_null_dists), alpha=0.8, color = '#175ac6')
    ax2.axvline(mpd_dist, color = 'red', lw = 3, ls = '--')
    ax2.set_xlabel("Mean pairwise distance " + r'$\mathrm{MPD^{ \left ( 3 \right ) }}$' , fontsize = 9)
    ax2.set_ylabel("Frequency", fontsize = 12)

    ax2.text(0.78, 0.94, r'$p_{\mathrm{MPD^{ \left ( 3 \right ) }}} = $' + str(round(p_value, 4)), fontsize = 8, ha='center', va='center', transform=ax2.transAxes)
    ax2.text(0.78, 0.87, r'$z_{\mathrm{MPD^{ \left ( 3 \right ) }}} = $' + str(round(z_score, 2)), fontsize = 8, ha='center', va='center', transform=ax2.transAxes)

    ax2.text(0.78, 0.76, r'$p_{ \tilde{L}_{1} } \ll 10^{-4}$', fontsize = 8, ha='center', va='center', transform=ax2.transAxes)
    ax2.text(0.78, 0.69, r'$z_{ \tilde{L}_{1} } = $' + str(round(z_score_eigen, 2)), fontsize = 8, ha='center', va='center', transform=ax2.transAxes)



    # plot hist and get p value
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    df_N = pd.read_csv(mydir + '/data/Tenaillon_et_al/power_corr_sample_size.txt', sep='\t')
    ax3.errorbar(df_N.N.values, df_N.Power.values, yerr = [df_N.Power.values-df_N.Power_975.values, df_N.Power_025.values - df_N.Power.values ], \
        fmt = 'o', alpha = 1, \
        barsabove = True, marker = '.', mfc = '#175ac6', mec = 'k', c = 'k', zorder=1)
    ax3.set_xlabel('Number of replicate populations', fontsize = 10)
    ax3.set_ylabel('Statistical power', fontsize = 12)
    ax3.axhline(0.05, color = 'dimgrey', lw = 2, ls = '--')
    ax3.set_xlim([-5,115])
    ax3.set_ylim([-0.1,1.1])


    ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax4.errorbar(df_N.N.values, df_N.z_score_mean.values, yerr = [df_N.z_score_mean.values-df_N.z_score_975.values, df_N.z_score_025.values - df_N.z_score_mean.values ], \
        fmt = 'o', alpha = 1, \
        barsabove = True, marker = '.', mfc = '#175ac6', mec = 'k', c = 'k', zorder=1)
    ax4.set_xlabel('Number of replicate populations', fontsize = 10)
    ax4.set_ylabel('Standardized ' + r'$\mathrm{MPD^{ \left ( 3 \right ) }}$', fontsize = 12)
    ax4.set_xlim([-5, 115])
    ax4.set_ylim([-0.7, 3.3])

    ax4.axhline(0, color = 'dimgrey', lw = 2, ls = '--', zorder=1)
    ax4.axhline(1, color = 'black', lw = 2, ls = ':', zorder=1)
    ax4.axhline(2, color = 'black', lw = 2, ls = ':', zorder=1)
    ax4.axhline(3, color = 'black', lw = 2, ls = ':', zorder=1)


    ax1.text(-0.1, 1.05, "a", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax1.transAxes)
    ax2.text(-0.1, 1.05, "b", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax2.transAxes)
    ax3.text(-0.1, 1.05, "c", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax3.transAxes)
    ax4.text(-0.1, 1.05, "d", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax4.transAxes)

    #fig.subplots_adjust(hspace=0)

    plt.tight_layout()
    fig.savefig(mydir + '/figs/tenaillon_corr.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()










def tenaillon_cov_PCA_fig(iter=10000):
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

    # get null data
    mpd_dist = pt.get_mean_pairwise_euc_distance(df_out)
    mpd_null_dists = []
    ks = range(2, 11)
    ch_scores = []
    ch_null_scores = []
    variance_ratios_null = []
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
        pca_i = PCA()
        df_out_i = pca_i.fit_transform(X_i)
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
    p_value = (len(mpd_greater) +1) / (iter+1)
    z_score = (mpd_dist - np.mean(mpd_null_dists)) / np.std(mpd_null_dists)
    print("p-value = " +  str(p_value))
    print("z-value = " +  str(z_score))


    fig = plt.figure(figsize = (6, 8))
    fig.tight_layout(pad = 2.8)
    # Scatterplot on main ax
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1)
    ax1.axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)
    ax1.axvline(x=0, color='k', linestyle=':', alpha = 0.8, zorder=2)

    df_pops_ESCRE1901 = df.loc[df['ESCRE1901'] > 0].index.tolist()
    df_pops_ECB_01992 = df.loc[df['ECB_01992'] > 0].index.tolist()

    #df_pops_rpoB = df.loc[df['rpoB'] > 0].index.tolist()
    #df_pops_ybaL = df.loc[df['ybaL'] > 0].index.tolist()
    #df_pops_ESCRE1901 = df.loc[df['ESCRE1901'] > 0].index.tolist()
    #df_pops_ECB_01992 = df.loc[df['ECB_01992'] > 0].index.tolist()

    df_out_labelled = pd.DataFrame(data=df_out, index=df.index.values)
    df_pops_ESCRE1901_pc = df_out_labelled.loc[ df_pops_ESCRE1901 , : ]
    df_pops_ECB_01992_pc = df_out_labelled.loc[ df_pops_ECB_01992 , : ]

    df_out_labelled_no_clsust = df_out_labelled.drop(df_pops_ESCRE1901+df_pops_ECB_01992)


    ax1.scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=3)

    ax1.set_xlim([-0.4,0.7])
    ax1.set_ylim([-0.4,0.7])
    print(pca.explained_variance_ratio_[0])
    ax1.set_xlabel('PC 1 (' + str(round(pca.explained_variance_ratio_[0]*100,2)) + '%)' , fontsize = 12)
    ax1.set_ylabel('PC 2 (' + str(round(pca.explained_variance_ratio_[1]*100,2)) + '%)' , fontsize = 12)

    ax1.scatter(df_pops_ESCRE1901_pc.values[:,0], df_pops_ESCRE1901_pc.values[:,1], marker = "o", edgecolors='#244162',  c = 'r', alpha = 0.4, s = 60, zorder=4)
    ax1.scatter(df_pops_ECB_01992_pc.values[:,0], df_pops_ECB_01992_pc.values[:,1], marker = "o", edgecolors='#244162',  c = 'purple', alpha = 0.4, s = 60, zorder=4)

    ax1.scatter(df_out_labelled_no_clsust.values[:,0], df_out_labelled_no_clsust.values[:,1], marker = "o", edgecolors='#244162',  c = '#175ac6', alpha = 0.4, s = 60, zorder=4)

    ax1.text(0.7,0.8,r'$n_{\mathrm{ESCRE1901}}=$' + str(len(df_pops_ESCRE1901)), fontsize=11, color='r', ha='center', va='center', transform=ax1.transAxes  )
    ax1.text(0.7,0.7,r'$n_{\mathrm{ECB \_ 01992 }} = $' + str(len(df_pops_ECB_01992)), fontsize=11, color='purple', ha='center', va='center', transform=ax1.transAxes)

    ax2 = plt.subplot2grid((3, 2), (0, 1), colspan=1)
    ax2.hist(mpd_null_dists, bins=30, weights=np.zeros_like(mpd_null_dists) + 1. / len(mpd_null_dists), alpha=0.8, color = '#175ac6')
    ax2.axvline(mpd_dist, color = 'red', lw = 3, ls = '--')
    ax2.set_xlabel("Mean pairwise distance " + r'$\mathrm{MPD^{ \left ( 3 \right ) }}$' , fontsize = 9)
    ax2.set_ylabel("Frequency", fontsize = 12)

    ax2.text(0.16, 0.9, r'$p = $' + str(round(p_value, 4)), fontsize = 8, ha='center', va='center', transform=ax2.transAxes)
    ax2.text(0.18, 0.83, r'$z_{\mathrm{MPD^{ \left ( 3 \right ) }}} = $' + str(round(z_score, 2)), fontsize = 8, ha='center', va='center', transform=ax2.transAxes)


    # plot hist and get p value
    ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=1)
    df_N = pd.read_csv(mydir + '/data/Tenaillon_et_al/power_cov_sample_size.txt', sep='\t')
    ax3.errorbar(df_N.N.values, df_N.Power.values, yerr = [df_N.Power.values-df_N.Power_975.values, df_N.Power_025.values - df_N.Power.values ], \
        fmt = 'o', alpha = 1, \
        barsabove = True, marker = '.', mfc = '#175ac6', mec = 'k', c = 'k', zorder=1)
    ax3.set_xlabel('Number of replicate populations', fontsize = 10)
    ax3.set_ylabel('Statistical power', fontsize = 12)
    ax3.axhline(0.05, color = 'dimgrey', lw = 2, ls = '--')
    ax3.set_xlim([-5,115])
    ax3.set_ylim([-0.1,1.1])


    ax4 = plt.subplot2grid((3, 2), (1, 1), colspan=1)
    ax4.errorbar(df_N.N.values, df_N.z_score_mean.values, yerr = [df_N.z_score_mean.values-df_N.z_score_975.values, df_N.z_score_025.values - df_N.z_score_mean.values ], \
        fmt = 'o', alpha = 1, \
        barsabove = True, marker = '.', mfc = '#175ac6', mec = 'k', c = 'k', zorder=1)
    ax4.set_xlabel('Number of replicate populations', fontsize = 10)
    ax4.set_ylabel('Standardized ' + r'$\mathrm{MPD^{ \left ( 3 \right ) }}$', fontsize = 12)
    ax4.set_xlim([-5, 115])
    ax4.set_ylim([-0.7, 3.3])

    ax4.axhline(0, color = 'dimgrey', lw = 2, ls = '--', zorder=1)
    ax4.axhline(1, color = 'black', lw = 2, ls = ':', zorder=1)
    ax4.axhline(2, color = 'black', lw = 2, ls = ':', zorder=1)
    ax4.axhline(3, color = 'black', lw = 2, ls = ':', zorder=1)


    ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=1)
    for ch_null_score in ch_null_scores:
        ax5.plot(list(range(2, 11)), ch_null_score, linestyle='-', alpha=0.4, color='royalblue')
    ax5.plot(list(range(2, 11)), ch_scores, linestyle='--', marker='o', color='red')
    ax5.set_xlabel('Number of clusters', fontsize = 12)
    ax5.set_ylabel('Variance ratio criterion', fontsize = 10)

    ax5.set_xlim([0.5,11])


    ax6 = plt.subplot2grid((3, 2), (2, 1), colspan=1)
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

    jaccard_means = []
    jaccard_025s = []
    jaccard_975s = []
    i_list = []
    for i in sorted(jaccard_dict, key=lambda k: sum(jaccard_dict[k]) / len(jaccard_dict[k]), reverse=True):
        y = jaccard_dict[i]
        x = np.random.normal(int(i)+1, 0.04, len(y)).tolist()
        ax6.scatter(x, y, c='#175ac6', alpha = 0.05, zorder=1)
        y_mean = np.mean(y)

        bs_jaccard_list = []
        for m in range(10000):
            bs_jaccard_list.append(np.mean(np.random.choice(y, size=20, replace=True)))
        bs_jaccard_list.sort()
        bs_jaccard_975 = bs_jaccard_list[ int(0.975 * 10000) ]
        bs_jaccard_025 = bs_jaccard_list[ int(0.025 * 10000) ]

        jaccard_means.append(y_mean)
        jaccard_025s.append(bs_jaccard_025)
        jaccard_975s.append(bs_jaccard_975)
        i_list.append(int(i)+1)

    ax6.errorbar(np.asarray(i_list), np.asarray(jaccard_means), yerr = [np.asarray(jaccard_means)-np.asarray(jaccard_975s), np.asarray(jaccard_025s) - np.asarray(jaccard_means) ], \
        fmt = 'o', alpha = 1, \
        barsabove = True, marker = '.',
        mfc = 'k', mec = 'k', c = 'k', zorder=2)

    ax6.set_ylabel('Jaccard similarity index', fontsize = 10)
    ax6.set_xticklabels(['', 'Cluster 1', 'Cluster 2', 'Cluster 3'])
    ax6.set_ylim([-0.08,0.84])

    ax1.text(-0.1, 1.05, "a", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax1.transAxes)
    ax2.text(-0.1, 1.05, "b", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax2.transAxes)
    ax3.text(-0.1, 1.05, "c", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax3.transAxes)
    ax4.text(-0.1, 1.05, "d", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax4.transAxes)
    ax5.text(-0.1, 1.05, "e", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax5.transAxes)
    ax6.text(-0.1, 1.05, "f", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax6.transAxes)

    #fig.subplots_adjust(hspace=0)

    plt.tight_layout()
    fig.savefig(mydir + '/figs/tenaillon_cov.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
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

    df_pops_ESCRE1901 = df.loc[df['ESCRE1901'] > 0].index.values
    df_pops_ECB_01992 = df.loc[df['ECB_01992'] > 0].index.values

    PC_1_2_df_ESCRE1901 = PC_1_2_W_df.loc[ df_pops_ESCRE1901 , : ]
    PC_1_2_df_ECB_01992 = PC_1_2_W_df.loc[ df_pops_ECB_01992 , : ]

    PC_1_2_df_labelled_no_clsust = PC_1_2_W_df.drop(np.concatenate((df_pops_ESCRE1901, df_pops_ECB_01992), axis=0))

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
    axes[0].scatter(PC_1_2_df_ESCRE1901.PC1.values, PC_1_2_df_ESCRE1901['W (avg)'].values, \
        c='r', marker = 'o', s = 70, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)#, edgecolors='none')
    axes[0].scatter(PC_1_2_df_ECB_01992.PC1.values, PC_1_2_df_ECB_01992['W (avg)'].values, \
        c='purple', marker = 'o', s = 70, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)#, edgecolors='none')
    axes[0].scatter(PC_1_2_df_labelled_no_clsust.PC1.values, PC_1_2_df_labelled_no_clsust['W (avg)'].values, \
        c='#175ac6', marker = 'o', s = 70, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)#, edgecolors='none')
    axes[0].set_xlabel("PC 1", fontsize = 18)
    axes[0].set_ylabel("Relative fitness, " + r'$W$', fontsize = 18)
    axes[0].text(0.37, 1.75, r'$p \nless 0.05$', fontsize = 10)
    axes[0].set(adjustable='box-forced', aspect='equal')

    axes[1].scatter(PC_1_2_df_ESCRE1901.PC2.values, PC_1_2_df_ESCRE1901['W (avg)'].values, \
        c='r', marker = 'o', s = 70, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)#, edgecolors='none')
    axes[1].scatter(PC_1_2_df_ECB_01992.PC2.values, PC_1_2_df_ECB_01992['W (avg)'].values, \
        c='purple', marker = 'o', s = 70, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)#, edgecolors='none')
    axes[1].scatter(PC_1_2_df_labelled_no_clsust.PC2.values, PC_1_2_df_labelled_no_clsust['W (avg)'].values, \
        c='#175ac6', marker = 'o', s = 70, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)#, edgecolors='none'

    axes[1].set_xlabel("PC 2", fontsize = 18)
    axes[1].set_ylabel("Relative fitness, " + r'$W$', fontsize = 18)
    axes[1].text(0.37, 1.75, r'$p \nless 0.05$', fontsize = 10)
    axes[1].set(adjustable='box-forced', aspect='equal')

    axes[0].text(-0.1, 1.15, "a", fontsize=12, fontweight='bold', ha='center', va='center', transform=axes[0].transAxes)
    axes[1].text(-0.1, 1.15, "b", fontsize=12, fontweight='bold', ha='center', va='center', transform=axes[1].transAxes)


    fig.tight_layout()
    fig.savefig(mydir + '/figs/tenaillon_PC_fitness.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()








def plot_ltee_pca(k=5, iter=100):
    df_path = mydir + '/data/Good_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    to_keep = pt.complete_nonmutator_lines()
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

    #fig = plt.figure()
    fig = plt.figure(figsize = (8, 4))
    fig.tight_layout(pad = 2.8)
    ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1)

    ax1.axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)
    ax1.axvline(x=0, color='k', linestyle=':', alpha = 0.8, zorder=2)
    ax1.scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=3)
    for pop in pt.complete_nonmutator_lines():
        if 'm' in pop:
            ltee_label = 'Ara-' + pop[-1]
        else:
            ltee_label = 'Ara+' + pop[-1]
        pop_df_pca = df_pca[df_pca.index.str.contains(pop)]
        c_list = [ color_dict[int(x.split('_')[1])] for x in pop_df_pca.index.values]
        if  pt.nonmutator_shapes()[pop] == 'p2':
            size == 50
        else:
            size = 80
        plt.scatter(pop_df_pca.values[:,0], pop_df_pca.values[:,1], \
        c=c_list, cmap = cm.Blues, vmin=min(time_points_set), vmax=max(time_points_set), \
        marker = pt.nonmutator_shapes()[pop], s = size, edgecolors='#244162', linewidth = 0.6, \
        label = ltee_label, alpha=0.9, zorder=4)#, edgecolors='none')

    #ax2_divider = make_axes_locatable(ax1)
    #im2 = ax1.imshow([[1, 2], [3, 4]])

    # add an axes above the main axes.
    #cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
    #cb2 = colorbar()
    #c = plt.colorbar( orientation="horizontal")
    #c.ax.tick_params(labelsize=6)

    # change tick position to top. Tick position defaults to bottom and overlaps
    # the image.
    #cax2.xaxis.set_ticks_position("top")
    p2 = ax1.get_position().get_points().flatten()

    ax_cbar1 = fig.add_axes([p2[0], 1, p2[2]-p2[0], 0.05])
    c = plt.colorbar(cax=ax_cbar1, orientation='horizontal')
    c.ax.tick_params(labelsize=6)

    ax1.text(0.5, 1.17, "Generations", fontsize=10, ha='center', va='center', transform=ax1.transAxes)

    #c.set_label("Generations", size=10)
    ax1.legend(loc='upper right', fontsize = 8)
    ax1.set_xlim([-0.25,0.4])
    ax1.set_ylim([-0.35,0.4])

    ax1.set_xlabel('PC 1 (' + str(round(pca.explained_variance_ratio_[0]*100,2)) + '%)' , fontsize = 13)
    ax1.set_ylabel('PC 2 (' + str(round(pca.explained_variance_ratio_[1]*100,2)) + '%)' , fontsize = 13)

    # analyze full LTEE in PCA

    # mean pairsise distance
    mean_dist = []
    for tp in time_points_set:
        df_pca_tp = df_pca[df_pca.index.str.contains('_' + str(tp))]
        mean_dist.append(pt.get_mean_pairwise_euc_distance(df_pca_tp.values, k = k))

    mpd_null_dict = {}
    for tp in time_points_set:
        mpd_null_dict[tp] = []
    for i in range(iter):
        if i % 1000 == 0:
            print(i)
        df_iter_list = []
        for pop in pt.complete_nonmutator_lines():
            pop_df_pca_iter = df_pca[df_pca.index.str.contains(pop)]
            row_names = pop_df_pca_iter.index.values
            pop_df_pca_iter = pop_df_pca_iter.sample(frac=1).reset_index(drop=True)
            pop_df_pca_iter = pop_df_pca_iter.set_index(row_names)

            df_iter_list.append(pop_df_pca_iter)

        df_pca_iter = pd.concat(df_iter_list)
        # get MDP
        for tp in time_points_set:
            df_pca_iter_tp = df_pca_iter[df_pca_iter.index.str.contains('_' + str(tp))]
            mpd_null_dict[tp].append(pt.get_mean_pairwise_euc_distance(df_pca_iter_tp.values, k = k))

    mpd_025 = []
    mpd_975 = []
    for tp in time_points_set:
        mpd_null_time_list = mpd_null_dict[tp]
        mpd_null_time_list.sort()
        mpd_025.append(mpd_null_time_list[ int(iter*0.025) ])
        mpd_975.append(mpd_null_time_list[ int(iter*0.975) ])

    ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1)

    ax2.scatter(time_points_set, mean_dist, marker = "o", edgecolors='#244162', c = '#175ac6', alpha = 0.8, s = 60, zorder=5)
    ax2.set_xlabel("Generations", fontsize = 12)
    ax2.set_ylabel("Mean pairwise distance, " + r'$\mathrm{MPD}^{\left ( 5 \right )}$', fontsize = 12)
    ax2.axvline(x=10000, c='dimgrey', ls='--', zorder=2)

    ax2.axhline(y=np.mean(mpd_025), c='k', ls=':', zorder=3)
    ax2.axhline(y=np.mean(mpd_975), c='k', ls=':', zorder=4)

    labels = [item.get_text() for item in ax2.get_xticklabels()]
    ax2.set_xticklabels(['', '0', '', '20000', '', '40000', '', '60000'])

    #ax2.fill_between(time_points_set, mpd_975, mpd_025, facecolor='lightskyblue', alpha=0.6, zorder=1)

    ax1.text(0, 1.15, "a", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax1.transAxes)
    ax2.text(0, 1.15, "b", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax2.transAxes)


    fig.tight_layout()
    fig.savefig(mydir + '/figs/pca_ltee.png', format='png', bbox_inches = "tight", pad_inches = 0.4,  dpi = 600)
    plt.close()



def ltee_eigen(k=5):
    df_path = mydir + '/data/Good_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    to_keep = pt.complete_nonmutator_lines()
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

    fig = plt.figure()

    broken_stick = []
    for i in range(1, len(pca.explained_variance_ratio_) +1):
        broken_stick.append(   (sum(1 / np.arange(i, len(pca.explained_variance_) +1)) / len(pca.explained_variance_)) * 100   )
    broken_stick = np.asarray(broken_stick)
    broken_stick = broken_stick / sum(broken_stick)

    plt.plot(list(range(1, len(pca.explained_variance_ratio_)+1)), pca.explained_variance_ratio_, linestyle='--', marker='o', color='red', alpha=0.6, label='Observed')
    plt.plot(list(range(1, len(pca.explained_variance_ratio_)+1)), broken_stick, linestyle=':', alpha=0.7, color='#175ac6', label='Broken-stick', lw=2.5)
    plt.legend(loc='upper right', fontsize='large')
    plt.xlabel('Eigenvalue rank', fontsize = 14)
    plt.ylabel('Proportion of variance explained', fontsize = 14)

    fig.tight_layout()
    fig.savefig(mydir + '/figs/pca_ltee_eigen.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def treatment_fig(iter=1000, control_BF=False):

    df_turner_path = mydir + '/data/Turner_et_al/gene_by_pop.txt'
    df_turner = pd.read_csv(df_turner_path, sep = '\t', header = 'infer', index_col = 0)
    df_turner = df_turner.loc[:, (df_turner != 0).any(axis=0)]

    #turner_treats = ['high_carbon_large_bead', 'high_carbon_planktonic', 'low_carbon_large_bead', 'low_carbon_planktonic']
    df_turner_delta = cd.likelihood_matrix_array(df_turner.values, df_turner.columns.values, 'Turner_et_al').get_likelihood_matrix()

    hclb_idx = df_turner.index.str.startswith('high_carbon_large_bead')
    df_hclb = df_turner[hclb_idx]
    df_hclb = df_hclb.loc[:, (df_hclb != 0).any(axis=0)]

    hcpl_idx = df_turner.index.str.startswith('high_carbon_planktonic')
    df_hcpl = df_turner[hcpl_idx]
    df_hcpl = df_hcpl.loc[:, (df_hcpl != 0).any(axis=0)]

    lclb_idx = df_turner.index.str.startswith('low_carbon_large_bead')
    df_lclb = df_turner[lclb_idx]
    df_lclb = df_lclb.loc[:, (df_lclb != 0).any(axis=0)]

    lcpl_idx = df_turner.index.str.startswith('low_carbon_planktonic')
    df_lcpl = df_turner[lcpl_idx]
    df_lcpl = df_lcpl.loc[:, (df_lcpl != 0).any(axis=0)]


    N_turner = [df_hclb.shape[0], df_hcpl.shape[0], df_lclb.shape[0], df_lcpl.shape[0]]
    X = df_turner_delta/df_turner_delta.sum(axis=1)[:,None]
    X -= np.mean(X, axis = 0)
    pca = PCA()
    df_out = pca.fit_transform(X)
    df_out_k = df_out[:,:k_eval]

    hclb_mpd = pt.get_mean_pairwise_euc_distance(df_out_k[0: N_turner[0] ,  :], k=3)
    hcpl_mpd = pt.get_mean_pairwise_euc_distance(df_out_k[N_turner[0]: sum(N_turner[:2]),  :] , k=3)
    lclb_mpd = pt.get_mean_pairwise_euc_distance(df_out_k[sum(N_turner[:2]): sum(N_turner[:3]),:]  , k=3)
    lcpl_mpd = pt.get_mean_pairwise_euc_distance(df_out_k[sum(N_turner[:3]): sum(N_turner[:4]),:] , k=3)

    hclb_mpd_null = []
    hcpl_mpd_null = []
    lclb_mpd_null = []
    lcpl_mpd_null = []
    # get MPD for each group
    if control_BF == True:
        F = pt.get_F_2(df_out_k, N_turner )
    else:
        F = pt.get_F_1(df_out_k, N_turner )

    F_null = []
    for i in range(iter):
        if i %1000 == 0:
            print(i)
        np_hclb_i = pt.get_random_matrix(df_hclb.values)
        df_hclb_i = pd.DataFrame(data=np_hclb_i, index =df_hclb.index, columns=df_hclb.columns)

        np_hcpl_i = pt.get_random_matrix(df_hcpl.values)
        df_hcpl_i = pd.DataFrame(data=np_hcpl_i, index =df_hcpl.index, columns=df_hcpl.columns)

        np_lclb_i = pt.get_random_matrix(df_lclb.values)
        df_lclb_i = pd.DataFrame(data=np_lclb_i, index =df_lclb.index, columns=df_lclb.columns)

        np_lcpl_i = pt.get_random_matrix(df_lcpl.values)
        df_lcpl_i = pd.DataFrame(data=np_lcpl_i, index=df_lcpl.index, columns=df_lcpl.columns)

        df_list = [df_hclb_i, df_hcpl_i, df_lclb_i,df_lcpl_i]
        df_merge_i = pd.concat(df_list, axis=0, sort=True).fillna(0)
        df_merge_delta_i = cd.likelihood_matrix_array(df_merge_i.values, df_merge_i.columns.values, 'Turner_et_al').get_likelihood_matrix()
        X_i = df_merge_delta_i/df_merge_delta_i.sum(axis=1)[:,None]
        X_i -= np.mean(X_i, axis = 0)
        df_out_i = pca.fit_transform(X_i)
        df_out_k_i = df_out_i[:,:k_eval]

        if control_BF == True:
            F_i = pt.get_F_2(df_out_k_i, N_turner )
        else:
            F_i = pt.get_F_1(df_out_k_i, N_turner )

        F_null.append(F_i)

        hclb_mpd_null.append(pt.get_mean_pairwise_euc_distance(df_out_k_i[0: N_turner[0] ,  :], k=3))
        hcpl_mpd_null.append(pt.get_mean_pairwise_euc_distance(df_out_k_i[N_turner[0]: sum(N_turner[:2]),  :] , k=3))
        lclb_mpd_null.append(pt.get_mean_pairwise_euc_distance(df_out_k_i[sum(N_turner[:2]): sum(N_turner[:3]),:]  , k=3))
        lcpl_mpd_null.append(pt.get_mean_pairwise_euc_distance(df_out_k_i[sum(N_turner[:3]): sum(N_turner[:4]),:] , k=3))


    # same test for wannier et al
    df_ECNR2 = pd.read_csv(mydir + '/data/Wannier_et_al/ECNR2.1_mutation_table_clean.txt', sep = '\t', header = 'infer', index_col = 0)
    df_ECNR2 = df_ECNR2.loc[:, (df_ECNR2 != 0).any(axis=0)]
    df_C321_deltaA_early = pd.read_csv(mydir + '/data/Wannier_et_al/C321.deltaA.earlyfix_mutation_table_clean.txt', sep = '\t', header = 'infer', index_col = 0)
    df_C321_deltaA_early = df_C321_deltaA_early.loc[:, (df_C321_deltaA_early != 0).any(axis=0)]
    df_C321_deltaA = pd.read_csv(mydir + '/data/Wannier_et_al/C321.deltaA_mutation_table_clean.txt', sep = '\t', header = 'infer', index_col = 0)
    df_C321_deltaA = df_C321_deltaA.loc[:, (df_C321_deltaA != 0).any(axis=0)]
    df_C321 = pd.read_csv(mydir + '/data/Wannier_et_al/C321_mutation_table_clean.txt', sep = '\t', header = 'infer', index_col = 0)
    df_C321 = df_C321.loc[:, (df_C321 != 0).any(axis=0)]

    df_w_list = [df_ECNR2, df_C321_deltaA_early, df_C321_deltaA, df_C321]
    df_w = pd.concat(df_w_list, axis=0, sort=True).fillna(0)
    df_w_delta = cd.likelihood_matrix_array(df_w.values, df_w.columns.values, 'Wannier_et_al').get_likelihood_matrix()

    N_w = [df_ECNR2.shape[0], df_C321_deltaA_early.shape[0], df_C321_deltaA.shape[0], df_C321.shape[0]]
    X_w = df_w_delta/df_w_delta.sum(axis=1)[:,None]
    X_w -= np.mean(X_w, axis = 0)
    df_out_w = pca.fit_transform(X_w)
    df_out_w_k = df_out_w[:,:k_eval]

    ECNR2_mpd = pt.get_mean_pairwise_euc_distance(df_out_w_k[0: N_w[0] ,  :], k=3)
    C321_deltaA_early_mpd = pt.get_mean_pairwise_euc_distance(df_out_w_k[sum(N_w[:1]): sum(N_w[:2]),  :] , k=3)
    C321_deltaA_mpd = pt.get_mean_pairwise_euc_distance(df_out_w_k[sum(N_w[:2]): sum(N_w[:3]),:]  , k=3)
    C321_mpd = pt.get_mean_pairwise_euc_distance(df_out_w_k[sum(N_w[:3]): sum(N_w[:4]),:] , k=3)

    ECNR2_mpd_null = []
    C321_deltaA_early_mpd_null = []
    C321_deltaA_mpd_null = []
    C321_mpd_null = []

    if control_BF == True:
        F_w = pt.get_F_2(df_out_w_k, N_w )
    else:
        F_w = pt.get_F_1(df_out_w_k, N_w)
    F_w_null = []

    for i in range(iter):
        if i %1000 == 0:
            print(i)
        np_ECNR2_i = pt.get_random_matrix(df_ECNR2.values)
        df_ECNR2_i = pd.DataFrame(data=np_ECNR2_i, index =df_ECNR2.index, columns=df_ECNR2.columns)

        np_C321_deltaA_early_i = pt.get_random_matrix(df_C321_deltaA_early.values)
        df_C321_deltaA_early_i = pd.DataFrame(data=np_C321_deltaA_early_i, index =df_C321_deltaA_early.index, columns=df_C321_deltaA_early.columns)

        np_C321_deltaA_i = pt.get_random_matrix(df_C321_deltaA.values)
        df_C321_deltaA_i = pd.DataFrame(data=np_C321_deltaA_i, index =df_C321_deltaA.index, columns=df_C321_deltaA.columns)

        np_C321_i = pt.get_random_matrix(df_C321.values)
        df_C321_i = pd.DataFrame(data=np_C321_i, index=df_C321.index, columns=df_C321.columns)

        df_w_merge_i = pd.concat([df_ECNR2_i, df_C321_deltaA_early_i, df_C321_deltaA_i, df_C321_i], axis=0, sort=True).fillna(0)
        df_w_merge_delta_i = cd.likelihood_matrix_array(df_w_merge_i.values, df_w_merge_i.columns.values, 'Wannier_et_al').get_likelihood_matrix()
        X_w_i = df_w_merge_delta_i/df_w_merge_delta_i.sum(axis=1)[:,None]
        X_w_i -= np.mean(X_w_i, axis = 0)
        df_w_out_i = pca.fit_transform(X_w_i)
        df_w_out_k_i = df_w_out_i[:,:k_eval]

        if control_BF == True:
            F_i = pt.get_F_2(df_w_out_k_i, N_w )
        else:
            F_i = pt.get_F_1(df_w_out_k_i, N_w )

        F_w_null.append(F_i)

        ECNR2_mpd_null.append(pt.get_mean_pairwise_euc_distance(df_w_out_k_i[0: N_w[0] ,  :], k=3))
        C321_deltaA_early_mpd_null.append(pt.get_mean_pairwise_euc_distance(df_w_out_k_i[N_w[0]: sum(N_w[:2]),  :] , k=3))
        C321_deltaA_mpd_null.append(pt.get_mean_pairwise_euc_distance(df_w_out_k_i[sum(N_w[:2]): sum(N_w[:3]),:]  , k=3))
        C321_mpd_null.append(pt.get_mean_pairwise_euc_distance(df_w_out_k_i[sum(N_w[:3]): sum(N_w[:4]),:] , k=3))


    fig = plt.figure(figsize = (6, 6))
    fig.tight_layout(pad = 2.8)

    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    z_lclb_mpd = (lclb_mpd - np.mean(lclb_mpd_null)) / np.std(lclb_mpd_null)
    z_lcpl_mpd = (lcpl_mpd - np.mean(lcpl_mpd_null)) / np.std(lcpl_mpd_null)
    z_hclb_mpd = (hclb_mpd - np.mean(hclb_mpd_null)) / np.std(hclb_mpd_null)
    z_hcpl_mpd = (hcpl_mpd - np.mean(hcpl_mpd_null)) / np.std(hcpl_mpd_null)

    z_lclb_mpd_null = (lclb_mpd_null - np.mean(lclb_mpd_null)) / np.std(lclb_mpd_null)
    z_lcpl_mpd_null = (lcpl_mpd_null - np.mean(lcpl_mpd_null)) / np.std(lcpl_mpd_null)
    z_hclb_mpd_null = (hclb_mpd_null - np.mean(hclb_mpd_null)) / np.std(hclb_mpd_null)
    z_hcpl_mpd_null = (hcpl_mpd_null - np.mean(hcpl_mpd_null)) / np.std(hcpl_mpd_null)

    z_lclb_mpd_null.sort()
    z_lcpl_mpd_null.sort()
    z_hclb_mpd_null.sort()
    z_hcpl_mpd_null.sort()

    lclb_mpd_025 = z_lclb_mpd_null[int(iter * 0.025) ]
    lcpl_mpd_025 = z_lcpl_mpd_null[int(iter * 0.025) ]
    hclb_mpd_025 = z_hclb_mpd_null[int(iter * 0.025) ]
    hcpl_mpd_025 = z_hcpl_mpd_null[int(iter * 0.025) ]

    lclb_mpd_975 = z_lclb_mpd_null[int(iter * 0.975) ]
    lcpl_mpd_975 = z_lcpl_mpd_null[int(iter * 0.975) ]
    hclb_mpd_975 = z_hclb_mpd_null[int(iter * 0.975) ]
    hcpl_mpd_975 = z_hcpl_mpd_null[int(iter * 0.975) ]

    z_lclb_mpd_null_mean = np.mean(z_lclb_mpd_null)
    z_lcpl_mpd_null_mean = np.mean(z_lcpl_mpd_null)
    z_hclb_mpd_null_mean = np.mean(z_hclb_mpd_null)
    z_hcpl_mpd_null_mean = np.mean(z_hcpl_mpd_null)

    x1 = [z_lclb_mpd_null_mean,z_lcpl_mpd_null_mean,z_hclb_mpd_null_mean,z_hcpl_mpd_null_mean]
    y1 = [1,2,3,4]
    xerr1 = [ [z_lclb_mpd_null_mean - lclb_mpd_025, z_lcpl_mpd_null_mean - lcpl_mpd_025, z_hclb_mpd_null_mean - hclb_mpd_025, z_hcpl_mpd_null_mean - hcpl_mpd_025 ] ,
            [lclb_mpd_975 - z_lclb_mpd_null_mean, lcpl_mpd_975 - z_lcpl_mpd_null_mean, hclb_mpd_975 - z_hclb_mpd_null_mean, hcpl_mpd_975 -z_hcpl_mpd_null_mean ]]

    ax1.errorbar(x1, y1, xerr = xerr1, \
            fmt = 'o', alpha = 0.9, barsabove = True, marker = '.', \
            mfc = 'k', mec = 'k', c = 'k', zorder=1, ms=17)

    ax1.scatter(z_lclb_mpd, 1, c='dodgerblue', marker = 'o', s = 80, \
        edgecolors='#244162', linewidth = 0, alpha = 1, zorder=2)
    ax1.scatter(z_lcpl_mpd, 2, c='lightgreen', marker = 'o', s = 80, \
        edgecolors='#244162', linewidth = 0, alpha = 1, zorder=2)
    ax1.scatter(z_hclb_mpd, 3, c='mediumblue', marker = 'o', s = 80, \
        edgecolors='#244162', linewidth = 0, alpha = 1, zorder=2)
    ax1.scatter(z_hcpl_mpd, 4, c='darkgreen', marker = 'o', s = 80, \
        edgecolors='#244162', linewidth = 0, alpha = 1, zorder=2)

    latex_labels = ["", "Low-C, biofilm", "", "Low-C, planktonic", "", "High-C, biofilm", "", "High-C, planktonic"]

    #ax1.set_yticks(list(range(len(latex_labels))), latex_labels)
    ax1.set_yticklabels(latex_labels, fontsize=9, rotation=45)

    #ax1.set_yticks([1,2,3,4])
    ax1.set_xlabel("Standardized mean pairwise\ndistance, " + r'$\mathrm{MPD^{ \left ( 3 \right ) }}$', fontsize = 10)
    #ax1.set_ylabel("Frequency", fontsize = 7)
    ax1.set_xlim(-2.5, 2.5)

    if control_BF == True:
        xlabel_F = "Between vs. within-treatment\nvariation, " + r'$F_{2}$'
    else:
        xlabel_F = "Between vs. within-treatment\nvariation, " + r'$F_{1}$'



    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    ax2.hist(F_null, bins=30,  weights=np.zeros_like(F_null) + 1. / len(F_null), alpha=0.6, color = '#175ac6')
    ax2.set_xlabel(xlabel_F, fontsize = 10)
    ax2.axvline(F, color = 'red', lw = 2, ls = '--')
    ax2.set_ylabel("Frequency", fontsize = 10)


    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1)

    z_ECNR2_mpd = (ECNR2_mpd - np.mean(ECNR2_mpd_null)) / np.std(ECNR2_mpd_null)
    z_C321_deltaA_early_mpd = (C321_deltaA_early_mpd - np.mean(C321_deltaA_early_mpd_null)) / np.std(C321_deltaA_early_mpd_null)
    z_C321_deltaA_mpd = (C321_deltaA_mpd - np.mean(C321_deltaA_mpd_null)) / np.std(C321_deltaA_mpd_null)
    z_C321_mpd = (C321_mpd - np.mean(C321_mpd_null)) / np.std(C321_mpd_null)

    z_ECNR2_mpd_null = (ECNR2_mpd_null - np.mean(ECNR2_mpd_null)) / np.std(ECNR2_mpd_null)
    z_C321_deltaA_early_mpd_null = (C321_deltaA_early_mpd_null - np.mean(C321_deltaA_early_mpd_null)) / np.std(C321_deltaA_early_mpd_null)
    z_C321_deltaA_mpd_null = (C321_deltaA_mpd_null - np.mean(C321_deltaA_mpd_null)) / np.std(C321_deltaA_mpd_null)
    z_C321_mpd_null = (C321_mpd_null - np.mean(C321_mpd_null)) / np.std(C321_mpd_null)

    z_ECNR2_mpd_null.sort()
    z_C321_deltaA_early_mpd_null.sort()
    z_C321_deltaA_mpd_null.sort()
    z_C321_mpd_null.sort()

    z_ECNR2_mpd_025 = z_ECNR2_mpd_null[int(iter * 0.025) ]
    z_C321_deltaA_early_mpd_025 = z_C321_deltaA_early_mpd_null[int(iter * 0.025) ]
    z_C321_deltaA_mpd_025 = z_C321_deltaA_mpd_null[int(iter * 0.025) ]
    z_C321_mpd_025 = z_C321_mpd_null[int(iter * 0.025) ]

    z_ECNR2_mpd_975 = z_ECNR2_mpd_null[int(iter * 0.975) ]
    z_C321_deltaA_early_mpd_975 = z_C321_deltaA_early_mpd_null[int(iter * 0.975) ]
    z_C321_deltaA_mpd_975 = z_C321_deltaA_mpd_null[int(iter * 0.975) ]
    z_C321_mpd_975 = z_C321_mpd_null[int(iter * 0.975) ]

    z_ECNR2_mpd_null_mean = np.mean(z_ECNR2_mpd_null)
    z_C321_deltaA_early_mpd_null_mean = np.mean(z_C321_deltaA_early_mpd_null)
    z_C321_deltaA_mpd_null_mean = np.mean(z_C321_deltaA_mpd_null)
    z_C321_mpd_null_mean = np.mean(z_C321_mpd_null)

    x3 = [z_ECNR2_mpd_null_mean,z_C321_deltaA_early_mpd_null_mean,z_C321_deltaA_mpd_null_mean,z_C321_mpd_null_mean]
    y3 = [1,2,3,4]
    xerr3 = [ [z_ECNR2_mpd_null_mean - z_ECNR2_mpd_025,
                z_C321_deltaA_early_mpd_null_mean - z_C321_deltaA_early_mpd_025,
                z_C321_deltaA_mpd_null_mean - z_C321_deltaA_mpd_025,
                z_C321_mpd_null_mean - z_C321_mpd_025 ] ,
            [z_ECNR2_mpd_975 - z_ECNR2_mpd_null_mean,
            z_C321_deltaA_early_mpd_975 - z_C321_deltaA_early_mpd_null_mean,
            z_C321_deltaA_mpd_975 - z_C321_deltaA_mpd_null_mean,
            z_C321_mpd_975 -z_C321_mpd_null_mean ]]

    ax3.errorbar(x3, y3, xerr = xerr3, \
            fmt = 'o', alpha = 0.9, barsabove = True, marker = '.', \
            mfc = 'k', mec = 'k', c = 'k', zorder=1, ms=17)

    ax3.scatter(z_ECNR2_mpd, 1, c='cornflowerblue', marker = 'o', s = 80, \
        edgecolors='#244162', linewidth = 0, alpha = 1, zorder=2)
    ax3.scatter(z_C321_deltaA_early_mpd, 2, c='goldenrod', marker = 'o', s = 80, \
        edgecolors='#244162', linewidth = 0, alpha = 1, zorder=2)
    ax3.scatter(z_C321_deltaA_mpd, 3, c='firebrick', marker = 'o', s = 80, \
        edgecolors='#244162', linewidth = 0, alpha = 1, zorder=2)
    ax3.scatter(z_C321_mpd, 4, c='seagreen', marker = 'o', s = 80, \
        edgecolors='#244162', linewidth = 0, alpha = 1, zorder=2)


    latex_labels = ["",  r'$\mathrm{ECNR2}$', "", r'$\mathrm{C321.}\Delta\mathrm{A}$' + '-' + r'$\mathrm{v2}$', "", r'$\mathrm{C321.}\Delta\mathrm{A}$', "", r'$\mathrm{C321}$']
    ax3.set_yticklabels(latex_labels, fontsize=9, rotation=45)
    ax3.set_xlabel("Standardized mean pairwise\ndistance, " + r'$\mathrm{MPD^{ \left ( 3 \right ) }}$', fontsize = 10)
    #ax1.set_ylabel("Frequency", fontsize = 7)
    ax3.set_xlim(-1.5, 3.5)

    ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax4.hist(F_w_null, bins=30,  weights=np.zeros_like(F_w_null) + 1. / len(F_w_null), alpha=0.6, color = '#175ac6')
    ax4.set_xlabel(xlabel_F, fontsize = 10)
    ax4.axvline(F_w, color = 'red', lw = 2, ls = '--')
    ax4.set_ylabel("Frequency", fontsize = 10)


    p_F_ax2 = len([ x for x in F_null if x > F]) / len(F_null)
    p_F_ax4 = len([ x for x in F_w_null if x > F_w]) / len(F_w_null)

    ax2.text(0.8, 0.9, r'$p=$' + str(round(p_F_ax2 ,3) ), fontsize=9, ha='center', va='center', transform=ax2.transAxes)
    ax4.text(0.8, 0.9, r'$p=$' + str(round(p_F_ax4 ,3) ), fontsize=9, ha='center', va='center', transform=ax4.transAxes)

    ax1.text(-0.1, 1.1, "a", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax1.transAxes)
    ax2.text(-0.1, 1.1, "b", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax2.transAxes)
    ax3.text(-0.1, 1.1, "c", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax3.transAxes)
    ax4.text(-0.1, 1.1, "d", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax4.transAxes)

    #fig.text(-0.04, 0.5,'Frequency', fontsize = 19, va='center',rotation='vertical')
    fig.tight_layout()
    fig.savefig(mydir + '/figs/divergence_figs_F' + str(int(control_BF)+1) + '.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def treatment_eigen_figs(iter=1000):
    df_turner_path = mydir + '/data/Turner_et_al/gene_by_pop.txt'
    df_turner = pd.read_csv(df_turner_path, sep = '\t', header = 'infer', index_col = 0)
    df_turner = df_turner.loc[:, (df_turner != 0).any(axis=0)]
    df_turner_delta = cd.likelihood_matrix_array(df_turner.values, df_turner.columns.values, 'Turner_et_al').get_likelihood_matrix()

    X_turner = df_turner_delta/df_turner_delta.sum(axis=1)[:,None]
    X_turner -= np.mean(X_turner, axis = 0)
    pca_turner = PCA()
    df_out_turner = pca_turner.fit_transform(X_turner)

    df_ECNR2 = pd.read_csv(mydir + '/data/Wannier_et_al/ECNR2.1_mutation_table_clean.txt', sep = '\t', header = 'infer', index_col = 0)
    df_ECNR2 = df_ECNR2.loc[:, (df_ECNR2 != 0).any(axis=0)]
    df_C321_deltaA_early = pd.read_csv(mydir + '/data/Wannier_et_al/C321.deltaA.earlyfix_mutation_table_clean.txt', sep = '\t', header = 'infer', index_col = 0)
    df_C321_deltaA_early = df_C321_deltaA_early.loc[:, (df_C321_deltaA_early != 0).any(axis=0)]
    df_C321_deltaA = pd.read_csv(mydir + '/data/Wannier_et_al/C321.deltaA_mutation_table_clean.txt', sep = '\t', header = 'infer', index_col = 0)
    df_C321_deltaA = df_C321_deltaA.loc[:, (df_C321_deltaA != 0).any(axis=0)]
    df_C321 = pd.read_csv(mydir + '/data/Wannier_et_al/C321_mutation_table_clean.txt', sep = '\t', header = 'infer', index_col = 0)
    df_C321 = df_C321.loc[:, (df_C321 != 0).any(axis=0)]

    df_w_list = [df_ECNR2, df_C321_deltaA_early, df_C321_deltaA, df_C321]
    df_w = pd.concat(df_w_list, axis=0, sort=True).fillna(0)
    df_w_delta = cd.likelihood_matrix_array(df_w.values, df_w.columns.values, 'Wannier_et_al').get_likelihood_matrix()

    X_w = df_w_delta/df_w_delta.sum(axis=1)[:,None]
    X_w -= np.mean(X_w, axis = 0)
    pca_w = PCA()
    df_out_w = pca_w.fit_transform(X_w)


    broken_stick_turner = []
    for i in range(1, len(pca_turner.explained_variance_ratio_) +1):
        broken_stick_turner.append(   (sum(1 / np.arange(i, len(pca_turner.explained_variance_) +1)) / len(pca_turner.explained_variance_)) * 100   )
    broken_stick_turner = np.asarray(broken_stick_turner)
    broken_stick_turner = broken_stick_turner / sum(broken_stick_turner)

    broken_stick_w = []
    for i in range(1, len(pca_w.explained_variance_ratio_) +1):
        broken_stick_w.append(   (sum(1 / np.arange(i, len(pca_w.explained_variance_) +1)) / len(pca_w.explained_variance_)) * 100   )
    broken_stick_w = np.asarray(broken_stick_w)
    broken_stick_w = broken_stick_w / sum(broken_stick_w)


    fig = plt.figure(figsize = (6, 3))
    fig.tight_layout(pad = 2.8)


    ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1)
    ax1.plot(list(range(1, len(pca_turner.explained_variance_ratio_)+1)), pca_turner.explained_variance_ratio_, linestyle='--', marker='o', color='red', alpha=0.6, label='Observed')
    ax1.plot(list(range(1, len(pca_turner.explained_variance_ratio_)+1)), broken_stick_turner, linestyle=':', alpha=0.7, color='#175ac6', label='Broken-stick', lw=2.5)
    #ax1.legend(loc='upper right', fontsize='large')
    ax1.set_xlabel('Eigenvalue rank', fontsize = 12)
    ax1.set_ylabel('Proportion of variance explained', fontsize = 11)
    ax1.set_title('Turner et al. dataset', fontsize = 13)

    ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1)
    ax2.plot(list(range(1, len(pca_w.explained_variance_ratio_)+1)), pca_w.explained_variance_ratio_, linestyle='--', marker='o', color='red', alpha=0.6, label='Observed')
    ax2.plot(list(range(1, len(pca_w.explained_variance_ratio_)+1)), broken_stick_w, linestyle=':', alpha=0.7, color='#175ac6', label='Broken-stick', lw=2.5)
    ax2.legend(loc='upper right', fontsize='large')
    ax2.set_xlabel('Eigenvalue rank', fontsize = 12)
    ax2.set_ylabel('Proportion of variance explained', fontsize = 11)
    ax2.set_title('Wannier et al. dataset', fontsize = 13)

    ax1.text(-0.1, 1.1, "a", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax1.transAxes)
    ax2.text(-0.1, 1.1, "b", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax2.transAxes)


    fig.tight_layout()
    fig.savefig(mydir + '/figs/pca_divergence_eigen.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()





def power_N_G_fig(alpha = 0.05):
    df_N = pd.read_csv(mydir + '/data/simulations/cov_ba_ntwrk_N.txt', sep='\t')
    df_G = pd.read_csv(mydir + '/data/simulations/cov_ba_ntwrk_G.txt', sep='\t')

    fig = plt.figure(figsize = (6, 6))
    fig.tight_layout(pad = 2.8)

    # Scatterplot on main ax
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    measures = ['Eig', 'MCD_k1', 'MCD_k3', 'MPD_k1', 'MPD_k3']
    colors = ['k', 'firebrick', 'orangered',  'dodgerblue', 'lightskyblue']
    labels = [r'$\tilde{L}_{1}$', r'$\mathrm{MCD}^{\left ( 1 \right )}$', r'$\mathrm{MCD}^{\left ( 3 \right )}$', r'$\mathrm{MPD}^{\left ( 1 \right )}$', r'$\mathrm{MPD}^{\left ( 3 \right )}$' ]
    for i, measure in enumerate(measures):
        df_i = df_N[ df_N['Method'] == measure ]
        color_i = colors[i]
        _jitter = df_i.N.values + np.random.normal(0, 0.003, len(df_i.N.values))
        ax1.errorbar(_jitter, df_i.Power.values, yerr = [df_i.Power.values-df_i.Power_975.values, df_i.Power_025.values - df_i.Power.values ], \
            fmt = 'o', alpha = 1, ms=20,\
            marker = '.', mfc = color_i,
            mec = color_i, c = 'k', zorder=2, label=labels[i])

    ax1.legend(loc='upper left')
    ax1.set_xlabel('Number of replicate populations', fontsize = 10)
    ax1.set_ylabel("Statistical power\n" +r'$ \mathrm{P}\left ( \mathrm{reject} \; H_{0}   \mid H_{1} \;   \mathrm{is}\, \mathrm{true}, \, \alpha=0.05 \right ) $', fontsize = 10)
    ax1.set_xlim(2.5, 170)
    ax1.set_ylim(-0.02, 0.55)
    ax1.axhline(0.05, color = 'dimgrey', lw = 2, ls = '--', zorder=1)
    ax1.set_xscale('log', basex=2)


    # figure 2
    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    for i, measure in enumerate(measures):
        df_i = df_G[ df_G['Method'] == measure ]
        color_i = colors[i]
        x_jitter = df_i.G.values + np.random.normal(0, 0.003, len(df_i.G.values))
        ax2.errorbar(x_jitter, df_i.Power.values, yerr = [df_i.Power.values-df_i.Power_975.values, df_i.Power_025.values - df_i.Power.values ], \
            fmt = 'o', alpha = 1, ms=20,\
            marker = '.', mfc = color_i,
            mec = color_i, c = 'k', zorder=2, label=labels[i])


    ax2.set_xlabel('Number of genes', fontsize = 10)
    ax2.axhline(0.05, color = 'dimgrey', lw = 2, ls = '--', zorder=1)
    ax2.set_ylim(-0.02, 0.55)
    ax2.set_xlim(6.5, 170)
    ax2.set_xscale('log', basex=2)



    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    for i, measure in enumerate(measures):
        df_i = df_N[ df_N['Method'] == measure ]
        color_i = colors[i]
        _jitter = df_i.N.values + np.random.normal(0, 0.003, len(df_i.N.values))
        ax3.errorbar(_jitter, df_i.Z_mean.values, yerr = [df_i.Z_mean.values-df_i.Z_975.values, df_i.Z_025.values - df_i.Z_mean.values ], \
            fmt = 'o', alpha = 1, ms=20,\
            marker = '.', mfc = color_i,
            mec = color_i, c = 'k', zorder=2, label=labels[i])

    ax3.set_xlabel('Number of replicate populations', fontsize = 10)
    ax3.set_ylabel("Standard score", fontsize = 10)
    ax3.set_xlim(2.5, 170)
    ax3.set_ylim(-0.43, 1.6)
    ax3.set_xscale('log', basex=2)
    ax3.axhline(0, color = 'dimgrey', lw = 2, ls = '--', zorder=1)
    ax3.axhline(1, color = 'black', lw = 2, ls = ':', zorder=1)



    ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    for i, measure in enumerate(measures):
        df_i = df_G[ df_G['Method'] == measure ]
        color_i = colors[i]
        x_jitter = df_i.G.values + np.random.normal(0, 0.003, len(df_i.G.values))
        ax4.errorbar(x_jitter, df_i.Z_mean.values, yerr = [df_i.Z_mean.values-df_i.Z_975.values, df_i.Z_025.values - df_i.Z_mean.values ], \
            fmt = 'o', alpha = 1, ms=20,\
            marker = '.', mfc = color_i,
            mec = color_i, c = 'k', zorder=2, label=labels[i])

    ax4.set_xlabel('Number of genes', fontsize = 10)
    ax4.set_ylabel("Standard score", fontsize = 10)
    ax4.set_xlim(6.5, 170)
    ax4.set_ylim(-0.8, 1.8)
    ax4.set_xscale('log', basex=2)
    ax4.axhline(0, color = 'dimgrey', lw = 2, ls = '--', zorder=1)
    ax4.axhline(1, color = 'black', lw = 2, ls = ':', zorder=1)


    ax1.text(-0.1, 1.05, "a", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax1.transAxes)
    ax2.text(-0.1, 1.05, "b", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax2.transAxes)
    ax3.text(-0.1, 1.05, "c", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax3.transAxes)
    ax4.text(-0.1, 1.05, "d", fontsize=12, fontweight='bold', ha='center', va='center', transform=ax4.transAxes)



    plt.tight_layout()
    fig.savefig(mydir + '/figs/power_N_G.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()





def tenaillon_eigen_fig(k=5):
    df_path = mydir + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    # remove columns with all zeros
    gene_names = df.columns.tolist()
    df_delta = cd.likelihood_matrix_array(df, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()
    X = df_delta/df_delta.sum(axis=1)[:,None]
    X -= np.mean(X, axis = 0)
    pca = PCA()
    df_out = pca.fit_transform(X)

    fig = plt.figure()

    print(pca.explained_variance_ratio_[0])

    broken_stick = []
    for i in range(1, len(pca.explained_variance_ratio_) +1):
        broken_stick.append(   (sum(1 / np.arange(i, len(pca.explained_variance_) +1)) / len(pca.explained_variance_)) * 100   )
    broken_stick = np.asarray(broken_stick)
    broken_stick = broken_stick / sum(broken_stick)

    plt.plot(list(range(1, len(pca.explained_variance_ratio_)+1)), pca.explained_variance_ratio_, linestyle='--', marker='o', color='red', alpha=0.6, label='Observed')
    plt.plot(list(range(1, len(pca.explained_variance_ratio_)+1)), broken_stick, linestyle=':', alpha=0.7, color='#175ac6', label='Broken-stick', lw=2.5)
    plt.legend(loc='upper right', fontsize='large')
    plt.xlabel('Eigenvalue rank', fontsize = 14)
    plt.ylabel('Proportion of variance explained', fontsize = 14)

    fig.tight_layout()
    fig.savefig(mydir + '/figs/tenaillon_eigen.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()






def get_rescaled_loadings():

    df_path = mydir + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    # remove columns with all zeros
    gene_names = df.columns.tolist()
    df_delta = cd.likelihood_matrix_array(df, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()
    X_rel = df_delta/df_delta.sum(axis=1)[:,None]
    X = X_rel - np.mean(X_rel, axis = 0)
    pca_genes = PCA()

    df_out = pca_genes.fit(X)

    loadings = np.transpose(df_out.components_) * np.sqrt(df_out.explained_variance_)
    loadings = np.transpose(loadings)

    # loading = covariance with unit scaled component
    # rescaled loading = variable-component correlation

    max_1 = np.abs(loadings[0]).argmax()
    cov_1 = np.abs(loadings[0])[max_1]
    gene_1 = gene_names[max_1]
    corr_1 = cov_1 / np.std(X_rel[:, max_1])

    max_2 = np.abs(loadings[1]).argmax()
    cov_2 = np.abs(loadings[1])[max_2]
    gene_2 = gene_names[max_2]
    corr_2 = cov_2 / np.std(X_rel[:, max_2])


    print(gene_1, corr_1**2)
    print(gene_2, corr_2**2)



def ltee_time_block_mcd(k=2, iterations=10000):

    df_path = mydir + '/data/Good_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    to_keep = pt.complete_nonmutator_lines()
    df_nonmut = df[df.index.str.contains('|'.join( to_keep))]
    # remove columns with all zeros
    df_nonmut = df_nonmut.loc[:, (df_nonmut != 0).any(axis=0)]
    # get null for time steps
    time_steps = [10250, 20250, 30250, 40250, 50250, 60250]
    #time_steps = ['10250','60250']
    fig, ax = plt.subplots(figsize=(4,4))

    z_mcd_null_mean_list = []
    z_mcd_null_975_list = []
    z_mcd_null_025_list = []

    z_mcd_timestep_list = []

    for time_step in time_steps:
        df_nonmut_timestep = df_nonmut[df_nonmut.index.str.contains('_'+str(time_step))]
        # remove columns with all zeros
        df_nonmut_timestep = df_nonmut_timestep.loc[:, (df_nonmut_timestep != 0).any(axis=0)]
        df_nonmut_timestep_np = df_nonmut_timestep.values
        df_nonmut_timestep_sample_names = df_nonmut_timestep.index.tolist()
        df_nonmut_timestep_gene_names = df_nonmut_timestep.columns.tolist()

        df_timestep_delta = cd.likelihood_matrix_array(df_nonmut_timestep_np, df_nonmut_timestep_gene_names, 'Good_et_al').get_likelihood_matrix()
        X_timestep = df_timestep_delta/df_timestep_delta.sum(axis=1)[:,None]
        X_timestep -= np.mean(X_timestep, axis = 0)

        pca_timestep = PCA()
        df_timestep_out = pca_timestep.fit_transform(X_timestep)
        mcd_timestep = pt.get_mean_pairwise_euc_distance(df_timestep_out, k = k)

        mcd_null = []
        for i in range(iterations):
            df_nonmut_timestep_np_i = pt.get_random_matrix(df_nonmut_timestep_np)
            df_timestep_delta_i = cd.likelihood_matrix_array(df_nonmut_timestep_np_i, df_nonmut_timestep_gene_names, 'Good_et_al').get_likelihood_matrix()

            X_timestep_i = df_timestep_delta_i/df_timestep_delta_i.sum(axis=1)[:,None]
            X_timestep_i -= np.mean(X_timestep_i, axis = 0)

            pca_timestep_i = PCA()
            df_timestep_out_i = pca_timestep_i.fit_transform(X_timestep_i)
            mcd_timestep_i = pt.get_mean_pairwise_euc_distance(df_timestep_out_i, k = k)

            mcd_null.append(mcd_timestep_i)

        mcd_null = np.asarray(mcd_null)
        z_mcd_null = (mcd_null - np.mean(mcd_null)) / np.std(mcd_null)
        z_mcd_timestep = (mcd_timestep - np.mean(mcd_null)) / np.std(mcd_null)

        z_mcd_null = np.sort(z_mcd_null)
        z_mcd_null_mean = np.mean(mcd_null)
        z_mcd_null_975 = z_mcd_null[int(0.975*iterations)]
        z_mcd_null_025 = z_mcd_null[int(0.025*iterations)]

        z_mcd_null_975_list.append(z_mcd_null_975)
        z_mcd_null_mean_list.append(z_mcd_null_mean)
        z_mcd_null_025_list.append(z_mcd_null_025)

        z_mcd_timestep_list.append(mcd_timestep)


    ax.errorbar(np.asarray(time_steps), np.asarray(z_mcd_null_mean_list), yerr = [np.asarray(z_mcd_null_mean_list)-np.asarray(z_mcd_null_025_list), np.asarray(z_mcd_null_975_list)-np.asarray(z_mcd_null_mean_list) ], \
        fmt = 'o', alpha = 1, lw=2,\
        barsabove = True, marker = 'o', ms =10,  mfc = 'k', mec = 'k', c = 'k', zorder=2)

    ax.scatter(time_steps, z_mcd_timestep_list, c='#175ac6', marker = 'o', s = 110, \
        edgecolors='none', linewidth = 0.6, alpha = 0.9, zorder=3)

    ax.set_xlabel('Generations, '+r'$t$', fontsize = 10)
    ax.set_ylabel('Standardized '+  r'$\mathrm{MPD}^{\left ( 3 \right )}$' + '\nfor substitutions up to ' + r'$t$', fontsize = 10)
    #ax4.set_xlim(6.5, 170)
    ax.set_ylim(-2.7, 2.7)
    ax.xaxis.set_tick_params(labelsize=8)

    #ax4.axhline(0, color = 'dimgrey', lw = 2, ls = '--', zorder=1)
    #ax4.axhline(1, color = 'black', lw = 2, ls = ':', zorder=1)

    fig.tight_layout()
    fig.savefig(mydir + '/figs/ltee_time_block_mcd.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def plot_mean_vs_var():
    df_path = mydir + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    # remove columns with all zeros
    gene_names = df.columns.tolist()
    df_delta = cd.likelihood_matrix_array(df, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()
    means = []
    variances = []
    for gene_counts in df_delta.T:
        gene_counts = gene_counts[gene_counts>0]

        if len(gene_counts)>=3:
            means.append(np.mean(gene_counts))
            variances.append(np.var(gene_counts))

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(means), np.log10(variances))
    x_log10 = np.log10(np.logspace(-0.5, 1.15, num=1000, endpoint=True, base=10))

    # hypothetical slope of 2
    ratio = (slope - 2) / std_err
    pval = stats.t.sf(np.abs(ratio), len(means)-2)*2

    fig, ax = plt.subplots(figsize=(4,4))

    ax.scatter(means, variances, c='#175ac6', marker = 'o', s = 110, \
        edgecolors='none', linewidth = 0.6, alpha = 0.9, zorder=3)

    ax.plot(10**x_log10, 10**(intercept + (x_log10 * slope)), ls='--', c='k', lw=2, label = r'$b = $' + str(round(slope/2, 3))  + ' OLS regression' )
    ax.plot(10**x_log10, 10**(intercept + (x_log10 * 2)), ls='--', c='grey', lw=2, label = r'$b = 1$' + ", Poisson")

    ax.set_xlabel('Mean multiplicity ' + r'$\overline{m}_{i}$', fontsize = 12)
    ax.set_ylabel('Variance of multiplicity ' + r'$\sigma^{2}_{m_{i}}$', fontsize = 12)

    ax.set_xscale('log', basex=10)
    ax.set_yscale('log', basey=10)

    ax.set_xlim([0.2,20])
    ax.set_ylim([0.001,10])

    ax.legend(loc="upper left", fontsize=8)

    ax.text(0.15,0.75,r'$r^{2}=$' + str(round(r_value**2, 3 )), fontsize=9, color='k', ha='center', va='center', transform=ax.transAxes  )
    ax.text(0.15,0.7,r'$P=$' + str(round(pval, 4)), fontsize=9, color='k', ha='center', va='center', transform=ax.transAxes  )


    fig.tight_layout()
    fig.savefig(mydir + '/figs/plot_mean_vs_var.pdf', format='pdf', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




#ltee_time_block_mcd()

#get_rescaled_loadings()

#tenaillon_sig_multiplicity_fig()
#tenaillon_fitnes_fig()

#tenaillon_PCA_fig()
#plot_ltee_pca()

#power_N_G_fig()
#power_method_fig()

#treatment_fig(iter=10000, control_BF=False)
#treatment_fig(iter=10000, control_BF=True)

#treatment_eigen_figs()
#tenaillon_eigen_fig()
#ltee_eigen()


tenaillon_corr_PCA_fig()
#plot_mean_vs_var()
