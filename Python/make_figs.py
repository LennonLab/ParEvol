from __future__ import division
import math, os, re
import numpy as np
import pandas as pd
import parevol_tools as pt
import matplotlib.pyplot as plt
from matplotlib import cm, rc_context
import matplotlib.patches as mpatches
import clean_data as cd
from scipy.special import comb
from scipy import stats

from sklearn.decomposition import PCA, SparsePCA


def gene_space_fig():
    fig = plt.figure()

    L=20
    c1 = [5,5,5,5]
    c2 = [11,5,3,1]
    c3 = [17,1,1,1]
    omega_0 = []
    omega_1_1 = []
    omega_1_2 = []
    omega_1_3 = []
    ks = list(range(L+1))
    for k in ks:
        omega_0.append(int(comb(L, k)))
        omega_1_1.append(pt.comb_n_muts_k_genes(k, c1 ))
        omega_1_2.append(pt.comb_n_muts_k_genes(k, c2 ))
        omega_1_3.append(pt.comb_n_muts_k_genes(k, c3 ))

    plt.plot(ks, np.asarray(omega_1_1), linestyle='--', lw =2.2, color='#87CEEB', alpha = 0.7, label = r'$\mathcal{L} =  \left \{  5,5,5,5 \right \}$')
    plt.plot(ks, np.asarray(omega_1_2), linestyle='--', lw =2.2, color='#FFA500', alpha = 0.7, label = r'$\mathcal{L} =  \left \{  11,5,3,1 \right \}$')
    plt.plot(ks, np.asarray(omega_1_3), linestyle='--', lw =2.2, color='#FF6347', alpha = 0.7, label = r'$\mathcal{L} =  \left \{  17,1,1,1 \right \}$')



    plt.xlabel('Substitutions, ' + r'$k$', fontsize = 20)
    plt.ylabel('Number of evolutionary\noutcomes, ' + r'$\left | \mathcal{G}_{1}\left ( k \right )  \right | $', fontsize = 18)

    plt.legend(loc='upper left', fontsize=14)
    #plt.yscale("log")

    plt.tight_layout()
    fig_name = pt.get_path() + '/figs/gene_space.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def intro_fig():
    #https://github.com/miloharper/visualise-neural-network
    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    f1=7
    omega_font=10

    #plt.gca().add_patch(circle)
    ax1.set_xlim([0,1])
    ax1.set_ylim([0,1])

    ax1_points = [ [0.5, 0, '0000'],
                    [0.125, 0.25, '0001'],
                    [0.375, 0.25, '0010'],
                    [0.625, 0.25, '0100'],
                    [0.875, 0.25, '1000'],
                    [0.0835, 0.5, '0011'],
                    [0.0835 + 0.167, 0.5, '0101'],
                    [0.0835 + (0.167*2), 0.5, '1001'],
                    [0.0835 + (0.167*3), 0.5, '0110'],
                    [0.0835 + (0.167*4), 0.5, '1010'],
                    [0.0835 + (0.167*4), 0.5, '1010'],
                    [0.9165, 0.5, '1100'],
                    [0.125, 0.75, '0111'],
                    [0.375, 0.75, '1011'],
                    [0.625, 0.75, '1101'],
                    [0.875, 0.75, '1110'],
                    [0.5, 1, '1111']]

    for ax1_point in ax1_points:
        ax1.text(ax1_point[0], ax1_point[1], ax1_point[2],
                      ha="center",
                      size=f1,
                      transform=ax1.transAxes,
                      bbox=dict(boxstyle='round', fc="w", ec="k"))


    ax1.plot([0.5,0.125], [0, 0.25], 'k-')
    ax1.plot([0.5,0.375], [0, 0.25], 'k-')
    ax1.plot([0.5,0.625], [0, 0.25], 'k-')
    ax1.plot([0.5,0.875], [0, 0.25], 'k-')

    ax1.plot([0.125,0.0835], [0.25, 0.5], 'k-')
    ax1.plot([0.125,0.0835 + 0.167], [0.25, 0.5], 'k-')
    ax1.plot([0.125,0.0835 + (0.167*2)], [0.25, 0.5], 'k-')

    ax1.plot([0.375,0.0835], [0.25, 0.5], 'k-')
    ax1.plot([0.375,0.0835 + (0.167*3)], [0.25, 0.5], 'k-')
    ax1.plot([0.375,0.0835 + (0.167*4)], [0.25, 0.5], 'k-')

    ax1.plot([0.625,0.0835 + 0.167], [0.25, 0.5], 'k-')
    ax1.plot([0.625,0.0835 + (0.167*3)], [0.25, 0.5], 'k-')
    ax1.plot([0.625,0.9165], [0.25, 0.5], 'k-')

    ax1.plot([0.875,0.0835 + (0.167*2)], [0.25, 0.5], 'k-')
    ax1.plot([0.875,0.0835 + (0.167*4)], [0.25, 0.5], 'k-')
    ax1.plot([0.875,0.9165], [0.25, 0.5], 'k-')

    ax1.plot([0.0835,0.125], [0.5,0.75], 'k-')
    ax1.plot([0.0835,0.375], [0.5,0.75], 'k-')
    ax1.plot([0.0835 + 0.167, 0.125], [0.5,0.75], 'k-')
    ax1.plot([0.0835 + 0.167, 0.625], [0.5,0.75], 'k-')
    ax1.plot([0.0835 + (0.167*2), 0.375], [0.5,0.75], 'k-')
    ax1.plot([0.0835 + (0.167*2), 0.625], [0.5,0.75], 'k-')
    ax1.plot([0.0835 + (0.167*3), 0.125], [0.5,0.75], 'k-')
    ax1.plot([0.0835 + (0.167*3), 0.875], [0.5,0.75], 'k-')
    ax1.plot([0.0835 + (0.167*4), 0.375], [0.5,0.75], 'k-')
    ax1.plot([0.0835 + (0.167*4), 0.875], [0.5,0.75], 'k-')
    ax1.plot([0.9165, 0.625], [0.5,0.75], 'k-')
    ax1.plot([0.9165, 0.875], [0.5,0.75], 'k-')

    ax1.plot([0.125,0.5], [0.75, 1], 'k-')
    ax1.plot([0.375,0.5], [0.75, 1], 'k-')
    ax1.plot([0.625,0.5], [0.75, 1], 'k-')
    ax1.plot([0.875,0.5], [0.75, 1], 'k-')

    #ax1.text(-0.24, -0.01, r'$\Omega_{0}\left ( 0 \right )=1$', fontsize = omega_font)
    #ax1.text(-0.24, 0.24, r'$\Omega_{0}\left ( 1 \right )=4$', fontsize = omega_font)
    #ax1.text(-0.24, 0.49, r'$\Omega_{0}\left ( 2 \right )=6$', fontsize = omega_font)
    #ax1.text(-0.24, 0.74, r'$\Omega_{0}\left ( 3 \right )=4$', fontsize = omega_font)
    #ax1.text(-0.24, 0.99, r'$\Omega_{0}\left ( 4 \right )=1$', fontsize = omega_font)

    ax1.text(-0.36, -0.01, r'$\left | \mathcal{G}_{0}\left ( 0 \right )  \right | =1$', fontsize = omega_font)
    ax1.text(-0.36, 0.24, r'$\left | \mathcal{G}_{0}\left ( 1 \right )  \right | =4$', fontsize = omega_font)
    ax1.text(-0.36, 0.49, r'$\left | \mathcal{G}_{0}\left ( 2 \right )  \right | =6$', fontsize = omega_font)
    ax1.text(-0.36, 0.74, r'$\left | \mathcal{G}_{0}\left ( 3 \right )  \right | =4$', fontsize = omega_font)
    ax1.text(-0.36, 0.99, r'$\left | \mathcal{G}_{0}\left ( 4 \right )  \right | =1$', fontsize = omega_font)

    ax1.axis('off')

    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    ax2.set_xlim([0,1])
    ax2.set_ylim([0,1])
    y_offset = 0.004
    ax2.text(0.5, 0-y_offset, '       ',
                  ha="center",
                  size=f1+2,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.47, 0, '0',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.53, 0, '0',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))


    ax2.text(0.25, 0.25 -y_offset, ' '*6,
                  ha="center",
                  size=f1+2,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.22, 0.25, '0',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.28, 0.25, '1',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))


    ax2.text(0.75, 0.25 -y_offset, ' '*6,
                  ha="center",
                  size=f1+2,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.72, 0.25, '1',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.78, 0.25, '0',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))


    ax2.text(0.165, 0.5 -y_offset, ' '*6,
                  ha="center",
                  size=f1+2,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.135, 0.5, '0',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.195, 0.5, '2',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))

    ax2.text(0.5, 0.5 -y_offset, ' '*6,
                  ha="center",
                  size=f1+2,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.47, 0.5, '1',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.53, 0.5, '1',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))


    ax2.text(0.835, 0.5 -y_offset, ' '*6,
                  ha="center",
                  size=f1+2,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.805, 0.5, '2',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.865, 0.5, '0',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))



    ax2.text(0.25, 0.75 -y_offset, ' '*6,
                  ha="center",
                  size=f1+2,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.22, 0.75, '1',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.28, 0.75, '2',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))


    ax2.text(0.75, 0.75 -y_offset, ' '*6,
                  ha="center",
                  size=f1+2,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.72, 0.75, '2',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.78, 0.75, '1',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))


    ax2.text(0.5, 1-y_offset, '       ',
                  ha="center",
                  size=f1+2,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.47, 1, '0',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))
    ax2.text(0.53, 1, '0',
                  ha="center",
                  size=f1,
                  transform=ax2.transAxes,
                  bbox=dict(boxstyle='round', fc="w", ec="k"))

    ax2.plot([0.5,0.25], [0, 0.25], 'k-')
    ax2.plot([0.5,0.75], [0, 0.25], 'k-')
    ax2.plot([0.25,0.165], [0.25, 0.5], 'k-')
    ax2.plot([0.25,0.5], [0.25, 0.5], 'k-')
    ax2.plot([0.75,0.5], [0.25, 0.5], 'k-')
    ax2.plot([0.75,0.835], [0.25, 0.5], 'k-')

    ax2.plot([0.165,0.25], [0.5, 0.75], 'k-')
    ax2.plot([0.5,0.25], [0.5, 0.75], 'k-')
    ax2.plot([0.5,0.75], [0.5, 0.75], 'k-')
    ax2.plot([0.835,0.75], [0.5, 0.75], 'k-')
    ax2.plot([0.25,0.5], [0.75, 1], 'k-')
    ax2.plot([0.75,0.5], [0.75, 1], 'k-')

    #ax2.text(-0.16, -0.01, r'$\mathcal{G}\left ( 0 \right )=1$', fontsize = omega_font)
    #ax2.text(-0.16, 0.24, r'$\Omega_{1}\left ( 1 \right )=2$', fontsize = omega_font)
    #ax2.text(-0.16, 0.49, r'$\Omega_{1}\left ( 2 \right )=3$', fontsize = omega_font)
    #ax2.text(-0.16, 0.74, r'$\Omega_{1}\left ( 3 \right )=2$', fontsize = omega_font)
    #ax2.text(-0.16, 0.99, r'$\Omega_{1}\left ( 4 \right )=1$', fontsize = omega_font)

    ax2.text(-0.28, -0.01, r'$\left | \mathcal{G}_{1}\left ( 0 \right )  \right | =1$', fontsize = omega_font)
    ax2.text(-0.28, 0.24, r'$\left | \mathcal{G}_{1}\left ( 1 \right )  \right | =2$', fontsize = omega_font)
    ax2.text(-0.28, 0.49, r'$\left | \mathcal{G}_{1}\left ( 2 \right )  \right | =3$', fontsize = omega_font)
    ax2.text(-0.28, 0.74, r'$\left | \mathcal{G}_{1}\left ( 3 \right )  \right | =2$', fontsize = omega_font)
    ax2.text(-0.28, 0.99, r'$\left | \mathcal{G}_{1}\left ( 4 \right )  \right | =1$', fontsize = omega_font)

    #ax1.text(-0.26, 0.99, r'$\left | \mathcal{G}_{0}\left ( 4 \right )  \right | =1$', fontsize = omega_font)
    ax2.axis('off')

    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1)#, aspect='equal')

    L=20
    c1 = [5,5,5,5]
    c2 = [11,5,3,1]
    c3 = [17,1,1,1]
    omega_0 = []
    omega_1_1 = []
    omega_1_2 = []
    omega_1_3 = []
    ks = list(range(L+1))
    for k in ks:
        omega_0.append(int(comb(L, k)))
        omega_1_1.append(pt.comb_n_muts_k_genes(k, c1 ))
        omega_1_2.append(pt.comb_n_muts_k_genes(k, c2 ))
        omega_1_3.append(pt.comb_n_muts_k_genes(k, c3 ))

    plt.plot(ks, np.asarray(omega_1_1)/ np.asarray(omega_0), linestyle='--', lw =2.2, color='#87CEEB', alpha = 0.7, label = r'$\mathcal{L} =  \left \{  5,5,5,5 \right \}$')
    plt.plot(ks, np.asarray(omega_1_2)/ np.asarray(omega_0), linestyle='--', lw =2.2, color='#FFA500', alpha = 0.7, label = r'$\mathcal{L} =  \left \{  11,5,3,1 \right \}$')
    plt.plot(ks, np.asarray(omega_1_3)/ np.asarray(omega_0), linestyle='--', lw =2.2, color='#FF6347', alpha = 0.7, label = r'$\mathcal{L} =  \left \{  17,1,1,1 \right \}$')


    ax3.set_xlabel('Substitutions, ' + r'$k$', fontsize = 16)
    #ax3.set_ylabel(r'$\mathrm{log}_{10}   \left (    \Omega_{1} / \Omega_{0} \right )$', fontsize = 15)
    #ax3.set_ylabel(r'$\frac{\left | \mathcal{G}_{1}\left ( k \right )  \right | }{\left | \mathcal{G}_{0}\left ( k \right )  \right | }, \, \mathrm{log}_{10}$', fontsize = 15)
    ax3.set_ylabel(r'$\mathrm{log}_{10}   \left (   \frac{\left | \mathcal{G}_{1}\left ( k \right )  \right | }{\left | \mathcal{G}_{0}\left ( k \right )  \right | }\right )$', fontsize = 15)

    ax3.legend(loc='upper center', fontsize=7)
    ax3.set_yscale("log")


    ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    c0_ax4 = [20]
    c1_ax4 = [10,10]
    c2_ax4 = [5,5,5,5]
    c3_ax4 = [2,2,2,2,2,2,2,2,2,2]
    omega_0 = []
    omega_1_1_ax4 = []
    omega_1_2_ax4 = []
    omega_1_3_ax4 = []
    omega_1_0_ax4 = []
    ks = list(range(L+1))
    for k in ks:
        omega_0.append(int(comb(L, k)))
        omega_1_1_ax4.append(pt.comb_n_muts_k_genes(k, c1_ax4 ))
        omega_1_2_ax4.append(pt.comb_n_muts_k_genes(k, c2_ax4 ))
        omega_1_3_ax4.append(pt.comb_n_muts_k_genes(k, c3_ax4 ))

        omega_1_0_ax4.append(pt.comb_n_muts_k_genes(k, c0_ax4 ))

    plt.plot(ks, np.asarray(omega_1_1_ax4)/ np.asarray(omega_0), linestyle='--', lw =2.2, color='#FF6347', alpha = 0.7, label=r'$G=2$')
    plt.plot(ks, np.asarray(omega_1_2_ax4)/ np.asarray(omega_0), linestyle='--', lw =2.2, color='#FFA500', alpha = 0.7, label=r'$G=4$')
    plt.plot(ks, np.asarray(omega_1_3_ax4)/ np.asarray(omega_0), linestyle='--', lw =2.2, color='#87CEEB', alpha = 0.7, label=r'$G=10$')

    ax4.set_xlabel('Substitutions, ' + r'$k$', fontsize = 16)
    #ax4.set_ylabel(r'$ \mathrm{log}_{10}   \left (    \Omega_{1} / \Omega_{0} \right )$', fontsize = 15)
    ax4.set_ylabel(r'$\mathrm{log}_{10}   \left (   \frac{\left | \mathcal{G}_{1}\left ( k \right )  \right | }{\left | \mathcal{G}_{0}\left ( k \right )  \right | }\right )$', fontsize = 15)
    ax4.legend(loc='lower left', fontsize=7)
    ax4.set_yscale("log")
    plt.tight_layout()
    fig_name = pt.get_path() + '/figs/test_network.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()





def hist_tenaillon(iter2 = 10000, k = 3):
    df_path = os.path.expanduser("~/GitHub/ParEvol") + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_np = df.values
    gene_names = df.columns.values
    df_np_delta = cd.likelihood_matrix_array(df_np, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()
    #rank_mean = df_np_delta.stack().groupby(df_np_delta.rank(method='first').stack().astype(float)).mean()
    #df_np_delta_q_norm = df_np_delta.rank(method='min').stack().astype(float).map(rank_mean).unstack()
    df_np_delta = df_np_delta/df_np_delta.sum(axis=1)[:,None]
    X = pt.get_mean_center(df_np_delta)
    #X = (df_np_delta - np.mean(df_np_delta)) / np.std(df_np_delta)
    pca = PCA()
    pca_fit = pca.fit_transform(X)
    euc_dist = pt.get_mean_pairwise_euc_distance(pca_fit, k =2)
    euc_dists = []

    for i in range(iter2):
        if i %1000 ==0:
            print(i)
        df_np_i = pt.get_random_matrix(df_np)
        np.seterr(divide='ignore')
        df_np_i_delta = cd.likelihood_matrix_array(df_np_i, gene_names, 'Tenaillon_et_al').get_likelihood_matrix()
        df_np_i_delta = df_np_i_delta/df_np_i_delta.sum(axis=1)[:,None]
        X_i = pt.get_mean_center(df_np_i_delta)
        #X_i = (df_np_i_delta - np.mean(df_np_i_delta)) / np.std(df_np_i_delta)
        pca_fit_i = pca.fit_transform(X_i)
        euc_dists.append( pt.get_mean_pairwise_euc_distance(pca_fit_i, k =2) )

    fig = plt.figure()
    plt.hist(euc_dists,bins=30, weights=np.zeros_like(euc_dists) + 1. / len(euc_dists), alpha=0.8, color = '#175ac6')
    plt.axvline(euc_dist, color = 'red', lw = 3)
    plt.xlabel("Euclidean distance", fontsize = 16)
    plt.ylabel("Standardized mean \n centroid distance", fontsize = 14)
    plt.ylabel("Frequency", fontsize = 12)
    fig.tight_layout()
    fig.savefig(os.path.expanduser("~/GitHub/ParEvol") + '/figs/test_hist_mc.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def hist_tenaillon_multi(k = 3):
    df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_delta = pt.likelihood_matrix(df, 'Tenaillon_et_al').get_likelihood_matrix()
    X = pt.hellinger_transform(df_delta)
    pca = PCA()
    df_out = pca.fit_transform(X)

    df_null_path = pt.get_path() + '/data/Tenaillon_et_al/permute_PCA.txt'
    df_null = pd.read_csv(df_null_path, sep = '\t', header = 'infer', index_col = 0)

    mean_angle = pt.get_mean_angle(df_out, k = k)
    mcd = pt.get_mean_centroid_distance(df_out, k=k)
    #mean_length = pt.get_euclidean_distance(df_out, k=k)
    mean_dist = pt.get_mean_pairwise_euc_distance(df_out, k=k)
    x_stat = pt.get_x_stat(pca.explained_variance_[:-1])

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    ax1.axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)
    ax1.axvline(x=0, color='k', linestyle=':', alpha = 0.8, zorder=2)
    ax1.scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=3)
    ax1.scatter(df_out[:,0], df_out[:,1], marker = "o", edgecolors='#244162', c = '#175ac6', alpha = 0.4, s = 60, zorder=4)

    ax1.set_xlim([-0.75,0.75])
    ax1.set_ylim([-0.75,0.75])
    ax1.set_xlabel('PCA 1 (' + str(round(pca.explained_variance_ratio_[0],3)*100) + '%)' , fontsize = 14)
    ax1.set_ylabel('PCA 2 (' + str(round(pca.explained_variance_ratio_[1],3)*100) + '%)' , fontsize = 14)


    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    mcd_list = df_null.MCD.tolist()
    #ax2.hist(mcd_list, bins=30, histtype='stepfilled', normed=True, alpha=0.6, color='b')
    ax2.hist(mcd_list,bins=30, weights=np.zeros_like(mcd_list) + 1. / len(mcd_list), alpha=0.8, color = '#175ac6')
    ax2.axvline(mcd, color = 'red', lw = 3)
    ax2.set_xlabel("Mean centroid distance, " + r'$ \left \langle \delta_{c}  \right \rangle$', fontsize = 14)
    ax2.set_ylabel("Frequency", fontsize = 16)

    mcd_list.append(mcd)
    relative_position_mcd = sorted(mcd_list).index(mcd) / (len(mcd_list) -1)
    if relative_position_mcd > 0.5:
        p_score_mcd = 1 - relative_position_mcd
    else:
        p_score_mcd = relative_position_mcd
    print('mean centroid distance p-score = ' + str(round(p_score_mcd, 3)))
    ax2.text(0.366, 0.088, r'$p < 0.05$', fontsize = 10)

    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    delta_L_list = df_null.mean_dist.tolist()
    #ax3.hist(delta_L_list, bins=30, histtype='stepfilled', normed=True, alpha=0.6, color='b')
    ax3.hist(delta_L_list,bins=30, weights=np.zeros_like(delta_L_list) + 1. / len(delta_L_list), alpha=0.8, color = '#175ac6')
    ax3.axvline(mean_dist, color = 'red', lw = 3)
    ax3.set_xlabel("Mean pair-wise \n Euclidean distance, " + r'$   \left \langle   d \right  \rangle$', fontsize = 14)
    ax3.set_ylabel("Frequency", fontsize = 16)

    delta_L_list.append(mean_dist)
    relative_position_delta_L = sorted(delta_L_list).index(mean_dist) / (len(delta_L_list) -1)
    if relative_position_delta_L > 0.5:
        p_score_delta_L = 1 - relative_position_delta_L
    else:
        p_score_delta_L = relative_position_delta_L
    print('mean difference in distances p-score = ' + str(round(p_score_delta_L, 3)))
    ax3.text(0.50, 0.09, r'$p < 0.05$', fontsize = 10)



    ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax4_values = df_null.x_stat.values
    ax4_values = ax4_values[np.logical_not(np.isnan(ax4_values))]
    #ax4.hist(ax4_values, bins=30, histtype='stepfilled', normed=True, alpha=0.6, color='b')
    ax4.hist(ax4_values, bins=30, weights=np.zeros_like(ax4_values) + 1. / len(ax4_values), alpha=0.8, color = '#175ac6')
    print(np.mean(ax4_values))
    print(stats.mode(ax4_values))

    ax4.axvline(x_stat, color = 'red', lw = 3)
    ax4.set_xlabel(r'$F_{1}$', fontsize = 14)
    ax4.set_ylabel("Frequency", fontsize = 16)

    mean_angle_list = ax4_values.tolist()
    mean_angle_list.append(mean_angle)
    relative_position_angle = sorted(mean_angle_list).index(mean_angle) / (len(mean_angle_list) -1)
    print(x_stat)
    print( len([x for x in mean_angle_list if x > x_stat])/  sum(mean_angle_list)  )
    if relative_position_angle > 0.5:
        p_score_angle = 1 - relative_position_angle
    else:
        p_score_angle = relative_position_angle
    print('F_{1} statistic p-score = ' + str(round(p_score_angle, 3)))
    ax4.text(19.1, 0.09, r'$p \nless  0.05$', fontsize = 10)

    plt.tight_layout()
    fig_name = pt.get_path() + '/figs/fig1.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def fig2(alpha = 0.05, k = 3):
    df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_delta = pt.likelihood_matrix(df, 'Tenaillon_et_al').get_likelihood_matrix()
    X = pt.hellinger_transform(df_delta)
    pca = PCA()
    df_out = pca.fit_transform(X)
    mcd = pt.get_mean_centroid_distance(df_out, k = k)
    mean_angle = pt.get_mean_angle(df_out, k = k)
    mean_length = pt.get_euclidean_distance(df_out, k=k)

    df_sample_path = pt.get_path() + '/data/Tenaillon_et_al/sample_size_permute_PCA.txt'
    df_sample = pd.read_csv(df_sample_path, sep = '\t', header = 'infer')#, index_col = 0)
    sample_sizes = sorted(list(set(df_sample.Sample_size.tolist())))

    fig = plt.figure()
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((6, 1), (2, 0), rowspan=2)
    ax3 = plt.subplot2grid((6, 1), (4, 0), rowspan=2)
    ax1.axhline(mcd, color = 'darkgray', lw = 3, ls = '--', zorder = 1)
    ax2.axhline(mean_angle, color = 'darkgray', lw = 3, ls = '--', zorder = 1)
    ax3.axhline(mean_length, color = 'darkgray', lw = 3, ls = '--', zorder = 1)
    for sample_size in sample_sizes:
        df_sample_size = df_sample.loc[df_sample['Sample_size'] == sample_size]
        x_sample_size = df_sample_size.Sample_size.values
        y_sample_size_mcd = np.sort(df_sample_size.MCD.values)
        y_sample_size_mean_angle = np.sort(df_sample_size.mean_angle.values)
        y_sample_size_delta_L = np.sort(df_sample_size.delta_L.values)

        lower_ci_mcd = np.mean(y_sample_size_mcd) -    y_sample_size_mcd[int(len(y_sample_size_mcd) * alpha)]
        upper_ci_mcd = abs(np.mean(y_sample_size_mcd) -    y_sample_size_mcd[int(len(y_sample_size_mcd) * (1 - alpha) )])

        lower_ci_angle = np.mean(y_sample_size_mean_angle) -    y_sample_size_mean_angle[int(len(y_sample_size_mean_angle) * alpha)]
        upper_ci_angle = abs(np.mean(y_sample_size_mean_angle) -    y_sample_size_mean_angle[int(len(y_sample_size_mean_angle) * (1 - alpha) )])

        lower_ci_delta_L = np.mean(y_sample_size_delta_L) -    y_sample_size_delta_L[int(len(y_sample_size_delta_L) * alpha)]
        upper_ci_delta_L = abs(np.mean(y_sample_size_delta_L) -    y_sample_size_delta_L[int(len(y_sample_size_delta_L) * (1 - alpha) )])
        with rc_context(rc={'errorbar.capsize': 3}):
            ax1.errorbar(sample_size, np.mean(y_sample_size_mcd), yerr = [np.asarray([lower_ci_mcd]), np.asarray([upper_ci_mcd])], c = 'k', fmt='-o') #, xerr=0.2, yerr=0.4)
            ax2.errorbar(sample_size, np.mean(y_sample_size_mean_angle), yerr = [np.asarray([lower_ci_angle]), np.asarray([upper_ci_angle])], c = 'k', fmt='-o')
            ax3.errorbar(sample_size, np.mean(y_sample_size_delta_L), yerr = [np.asarray([lower_ci_delta_L]), np.asarray([upper_ci_delta_L])], c = 'k', fmt='-o')

    ax3.set_xlabel("Number of replicate populations", fontsize = 16)
    ax1.set_ylabel(r'$\left \langle \delta_{c}  \right \rangle$', fontsize = 14)
    ax2.set_ylabel(r'$\left \langle \theta \right \rangle$', fontsize = 14)
    ax3.set_ylabel(r'$\left \langle  \left | \Delta L \right |\right \rangle$', fontsize = 14)
    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/fig2.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def ltee_convergence(alpha = 0.05, k = 5):
    df_path = os.path.expanduser("~/GitHub/ParEvol") + '/data/Good_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    to_keep = pt.complete_nonmutator_lines()
    #to_keep.append('p5')
    to_keep.remove('p5')
    df_nonmut = df[df.index.str.contains('|'.join( to_keep))]
    # remove columns with all zeros
    df_nonmut = df_nonmut.loc[:, (df_nonmut != 0).any(axis=0)]
    gene_names = df_nonmut.columns.tolist()
    sample_names = df_nonmut.index.tolist()
    df_delta = cd.likelihood_matrix_array(df_nonmut, gene_names, 'Good_et_al').get_likelihood_matrix()
    df_delta = df_delta/df_delta.sum(axis=1)[:,None]
    X = pt.get_mean_center(df_delta)

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

    # get null distribution


    #fig = plt.figure()
    #plt.scatter(time_points_set, mean_dist, marker = "o", edgecolors='#244162', c = '#175ac6', alpha = 0.4, s = 60, zorder=4)

    #plt.xlabel("Time", fontsize = 14)
    #plt.ylabel("Mean euclidean distance", fontsize = 12)

    #plt.figure(1)
    #plt.subplot(313)
    #plt.errorbar(perm_gens, mean_L, yerr = [lower_ci_L, upper_ci_L], fmt = 'o', alpha = 0.5, \
    #    barsabove = True, marker = '.', mfc = 'k', mec = 'k', c = 'k', zorder=1)
    #plt.scatter(time_points_set, Ls, c='#175ac6', marker = 'o', s = 70, \
    #    edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)#, edgecolors='none')

    #for pop in to_keep:
    #    pop_df_pca = df_pca[df_pca.index.str.contains(pop)]
    #    c_list = [ color_dict[int(x.split('_')[1])] for x in pop_df_pca.index.values]
    #    if  pt.nonmutator_shapes()[pop] == 'p2':
    #        size == 50
    #    else:
    #        size = 80
    #    plt.scatter(pop_df_pca.values[:,0], pop_df_pca.values[:,1], \
    #        c=c_list, cmap = cm.Blues, vmin=min(time_points_set), vmax=max(time_points_set), \
    #        marker = pt.nonmutator_shapes()[pop], s = size, edgecolors='#244162', \
    #        linewidth = 0.6,  zorder=4, alpha=0.7)#, edgecolors='none')

    #c = plt.colorbar()
    #c.set_label("Generations", size=18)
    #plt.xlabel('PCA 1 (' + str(round(pca.explained_variance_ratio_[0],3)*100) + '%)' , fontsize = 16)
    #plt.ylabel('PCA 2 (' + str(round(pca.explained_variance_ratio_[1],3)*100) + '%)' , fontsize = 16)
    #plt.xlim([-0.4,0.4])
    #plt.ylim([-0.4,0.4])
    fig.tight_layout()
    fig.savefig(os.path.expanduser("~/GitHub/ParEvol") + '/figs/ltee_convergence.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_permutation(dataset = 'good', analysis = 'PCA', alpha = 0.05):
    df_path = pt.get_path() + '/data/Good_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    to_exclude = pt.complete_nonmutator_lines()
    to_exclude.append('p5')
    df_nonmut = df[df.index.str.contains('|'.join( to_exclude))]
    # remove columns with all zeros
    df_nonmut = df_nonmut.loc[:, (df_nonmut != 0).any(axis=0)]
    df_delta = pt.likelihood_matrix(df_nonmut, 'Good_et_al').get_likelihood_matrix()
    if analysis == 'PCA':
        X = pt.hellinger_transform(df_delta)
        pca = PCA()
        df_out = pca.fit_transform(X)
    elif analysis == 'cMDS':
        df_delta_bc = np.sqrt(pt.get_scipy_bray_curtis(df_delta.as_matrix()))
        df_out = pt.cmdscale(df_delta_bc)[0]

    time_points = [ int(x.split('_')[1]) for x in df_nonmut.index.values]
    time_points_set = sorted(list(set([ int(x.split('_')[1]) for x in df_nonmut.index.values])))

    df_rndm_delta_out = pd.DataFrame(data=df_out, index=df_delta.index)
    mcds = []
    for tp in time_points_set:
        df_rndm_delta_out_tp = df_rndm_delta_out[df_rndm_delta_out.index.str.contains('_' + str(tp))]
        mcds.append(pt.get_mean_pairwise_euc_distance(df_rndm_delta_out_tp.as_matrix(), k=3))

    mcd_perm_path = pt.get_path() + '/data/Good_et_al/permute_' + analysis + '.txt'
    mcd_perm = pd.read_csv(mcd_perm_path, sep = '\t', header = 'infer', index_col = 0)
    mcd_perm_x = np.sort(list(set(mcd_perm.Generation.tolist())))
    lower_ci = []
    upper_ci = []
    mean_mcds = []
    std_mcds = []
    lower_z_ci = []
    upper_z_ci = []
    for x in mcd_perm_x:
        mcd_perm_y = mcd_perm.loc[mcd_perm['Generation'] == x]
        mcd_perm_y_sort = np.sort(mcd_perm_y.mean_dist.tolist())
        mean_mcd_perm_y = np.mean(mcd_perm_y_sort)
        std_mcd_perm_y = np.std(mcd_perm_y_sort)
        mean_mcds.append(mean_mcd_perm_y)
        std_mcds.append(std_mcd_perm_y)
        lower_ci.append(mean_mcd_perm_y - mcd_perm_y_sort[int(len(mcd_perm_y_sort) * alpha)])
        upper_ci.append(abs(mean_mcd_perm_y - mcd_perm_y_sort[int(len(mcd_perm_y_sort) * (1 - alpha))]))
        # z-scores
        mcd_perm_y_sort_z = [ ((i - mean_mcd_perm_y) /  std_mcd_perm_y) for i in mcd_perm_y_sort]
        lower_z_ci.append(abs(mcd_perm_y_sort_z[int(len(mcd_perm_y_sort_z) * alpha)]))
        upper_z_ci.append(abs(mcd_perm_y_sort_z[int(len(mcd_perm_y_sort_z) * (1 - alpha))]))

    fig = plt.figure()

    plt.figure(1)
    plt.subplot(211)
    plt.errorbar(mcd_perm_x, mean_mcds, yerr = [lower_ci, upper_ci], fmt = 'o', alpha = 0.5, \
        barsabove = True, marker = '.', mfc = 'k', mec = 'k', c = 'k', zorder=1)
    plt.scatter(time_points_set, mcds, c='#175ac6', marker = 'o', s = 70, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)#, edgecolors='none')

    #plt.xlabel("Time (generations)", fontsize = 16)
    #plt.ylabel("Mean \n Euclidean distance", fontsize = 14)
    plt.ylabel("Mean pair-wise \n Euclidean \n distance, " + r'$   \left \langle   d \right  \rangle$', fontsize = 14)


    plt.figure(1)
    plt.subplot(212)
    plt.errorbar(mcd_perm_x, [0] * len(mcd_perm_x), yerr = [lower_z_ci, upper_z_ci], fmt = 'o', alpha = 0.5, \
        barsabove = True, marker = '.', mfc = 'k', mec = 'k', c = 'k', zorder=1)
    # zip mean, std, and measured values to make z-scores
    zip_list = list(zip(mean_mcds, std_mcds, mcds))
    z_scores = [((i[2] - i[0]) / i[1]) for i in zip_list ]
    plt.scatter(time_points_set, z_scores, c='#175ac6', marker = 'o', s = 70, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)#, edgecolors='none')
    plt.ylim(-2.2, 2.2)
    #plt.axhline(0, color = 'k', lw = 2, ls = '-')
    #plt.axhline(-1, color = 'dimgrey', lw = 2, ls = '--')
    #plt.axhline(-2, color = 'dimgrey', lw = 2, ls = ':')
    plt.xlabel("Time (generations)", fontsize = 16)

    plt.ylabel("Standardized mean \n pair-wise Euclidean \n distance, " + r'$   z_{\left \langle   d \right  \rangle}$', fontsize = 14)
    #plt.ylabel("Standardized mean \n Euclidean distance", fontsize = 14)

    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/permutation_scatter_good.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def tenaillon_fitness_hist():
    gene_by_pop_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
    gene_by_pop = pd.read_csv(gene_by_pop_path, sep = '\t', header = 'infer', index_col = 0)
    fitness_path = pt.get_path() + '/data/Tenaillon_et_al/fitness.csv'
    fitness = pd.read_csv(fitness_path, sep = ',', header = 'infer', index_col = 0)
    # select fitness values from lines that were sequenced
    fitness_subset = fitness.ix[gene_by_pop.index.values]
    fitness_np = fitness_subset['W (avg)'].values
    fitness_np = fitness_np[np.logical_not(np.isnan(fitness_np))]

    kde = pt.get_kde(fitness_np)

    fig = plt.figure()
    plt.plot(kde[0], kde[1])
    plt.xlabel("Fitness", fontsize = 18)
    plt.ylabel("Frequency", fontsize = 18)
    fig.tight_layout()
    plot_path = pt.get_path() + '/figs/tenaillon_fitness.png'
    fig.savefig(plot_path, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def test_pca_regression():
    gene_by_pop_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
    gene_by_pop = pd.read_csv(gene_by_pop_path, sep = '\t', header = 'infer', index_col = 0)
    fitness_path = pt.get_path() + '/data/Tenaillon_et_al/fitness.csv'
    fitness = pd.read_csv(fitness_path, sep = ',', header = 'infer', index_col = 0)
    # select fitness values from lines that were sequenced
    # this also orders the observations
    fitness_subset = fitness.ix[gene_by_pop.index.values]
    fitness_np = fitness_subset['W (avg)'].values
    gene_by_pop_delta = pt.likelihood_matrix(gene_by_pop, 'Tenaillon_et_al').get_likelihood_matrix()
    X = pt.hellinger_transform(gene_by_pop_delta)
    pca = PCA()
    df_out = pca.fit_transform(X)
    pca1 = df_out[:,0]

    clean_zip = [i for i in list(zip(pca1, fitness_np)) if (math.isnan(i[0])==False ) and (math.isnan(i[1])==False )]
    x = np.asarray([i[0] for i in clean_zip])
    y = np.asarray([i[1] for i in clean_zip])

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    print("slope is " + str(slope))
    print("r2-value is " + str(r_value **2))
    print("p-value is " + str(p_value))

    predict_y = intercept + (slope * x)
    pred_error = y - predict_y
    degrees_of_freedom = len(x) - 2
    residual_std_error = np.sqrt(np.sum(pred_error**2) / degrees_of_freedom)

    fig = plt.figure()
    plt.scatter(x, y, c='#175ac6', marker = 'o', s = 70, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)#, edgecolors='none')
    plt.plot(x, predict_y, 'k-')
    plt.xlabel("PCA 1", fontsize = 18)
    plt.ylabel("Fitness", fontsize = 18)
    fig.tight_layout()
    plot_path = pt.get_path() + '/figs/tenaillon_pcr.png'
    fig.savefig(plot_path, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def power_figs(alpha = 0.05):
    df = pd.read_csv(pt.get_path() + '/data/simulations/cov_ba_ntwrk_ev.txt', sep='\t')
    fig = plt.figure()
    covs = [0.05,0.1,0.15,0.2]
    measures = ['euc_percent', 'eig_percent', 'mcd_percent_k1', 'mcd_percent_k3']
    colors = ['powderblue', 'skyblue', 'royalblue', 'blue', 'navy']
    labels = ['euclidean distance', 'eigenanalysis', 'mcd 1', 'mcd 1-3']

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
    plt.legend(loc='lower right')
    plt.xlabel('Covariance', fontsize = 16)
    plt.ylabel(r'$ \mathrm{P}\left ( \mathrm{reject} \; H_{0}   \mid H_{1} \;   \mathrm{is}\, \mathrm{true}, \, \alpha=0.05 \right ) $', fontsize = 16)
    #plt.xlim(-0.02, 1.02)
    #plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    fig_name = pt.get_path() + '/figs/power_method.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def tenaillon_p_N(alpha = 0.05, bootstraps=10000, bootstrap_sample = 100):
    fig = plt.figure()
    df = pd.read_csv(pt.get_path() + '/data/Tenaillon_et_al/sample_size_sim.txt', sep='\t')
    Ns = list(set(df.N.values))
    for N in Ns:
        print(N)
        N_dist = df.loc[df['N'] == N].dist_percent.values
        bootstrap_power = []
        for bootstrap in range(bootstraps):
            p_sig = [p_i for p_i in np.random.choice(N_dist, size = bootstrap_sample) if p_i >= (1-alpha)]
            bootstrap_power.append(len(p_sig) / bootstrap_sample)
        bootstrap_power = np.sort(bootstrap_power)
        N_power = len([p_i for p_i in N_dist if p_i >= (1- alpha)]) / len(N_dist)
        lower_ci = bootstrap_power[int(len(bootstrap_power) * 0.05)]
        upper_ci = bootstrap_power[  len(bootstrap_power) -  int(len(bootstrap_power) * 0.05)]
        plt.errorbar(N, N_power, yerr = [np.asarray([N_power-upper_ci]), np.asarray([lower_ci - N_power])], fmt = 'o', alpha = 0.5, \
            barsabove = True, marker = '.', mfc = 'k', mec = 'k', c = 'k', zorder=1)
        plt.scatter(N, N_power, c='#175ac6', marker = 'o', s = 70, \
            edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)
    plt.xlabel('Number of replicate populations', fontsize = 16)
    plt.ylabel('Empirical power', fontsize = 16)
    plt.axhline(0.05, color = 'dimgrey', lw = 2, ls = '--')
    plt.tight_layout()
    fig_name = pt.get_path() + '/figs/tenaillon_N.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def poisson_power_N(alpha = 0.05):
    fig = plt.figure()
    df = pd.read_csv(pt.get_path() + '/data/simulations/ba_cov_N_sims.txt', sep='\t')
    covs = np.sort(list(set(df.Cov.values)))
    Ns = np.sort(list(set(df.N.values)))
    colors = ['powderblue',  'royalblue', 'navy']
    for i, cov in enumerate(covs):
        powers = []
        for j, N in enumerate(Ns):
            df_cov = df[ (df['Cov'] == cov) & (df['N'] == N) ]
            p = df_cov['dist_percent'].values
            #p = df_i[ (df_i['N_genes_sample'] == gene_shuffle) ].p.tolist()
            p_sig = [p_i for p_i in p if p_i >= (1-alpha)]
            powers.append(len(p_sig) / len(p))
        plt.plot(np.asarray(Ns), np.asarray(powers), linestyle='--', marker='o', color=colors[i], label=r'$\mathrm{cov}=$' + str(cov))

    plt.tight_layout()
    plt.legend(loc='upper left', fontsize=14)
    plt.xlabel('Number of replicate populations, '+ r'$\mathrm{log}_{2}$', fontsize = 16)
    plt.xscale('log', basex=2)
    plt.axhline(0.05, color = 'dimgrey', lw = 2, ls = '--')
    plt.ylabel(r'$ \mathrm{P}\left ( \mathrm{reject} \; H_{0}   \mid H_{1} \;   \mathrm{is}\, \mathrm{true}, \, \alpha=0.05 \right ) $', fontsize = 16)
    fig_name = pt.get_path() + '/figs/poisson_power_N.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def poisson_power_G(alpha = 0.05):
    fig = plt.figure()
    df = pd.read_csv(pt.get_path() + '/data/simulations/ba_cov_G_sims.txt', sep='\t')
    covs = np.sort(list(set(df.Cov.values)))
    Ns = np.sort(list(set(df.G.values)))
    colors = ['powderblue',  'royalblue', 'navy']
    for i, cov in enumerate(covs):
        powers = []
        for j, N in enumerate(Ns):
            df_cov = df[ (df['Cov'] == cov) & (df['G'] == N) ]
            p = df_cov['dist_percent'].values
            #p = df_i[ (df_i['N_genes_sample'] == gene_shuffle) ].p.tolist()
            p_sig = [p_i for p_i in p if p_i >= (1-alpha)]
            powers.append(len(p_sig) / len(p))
        plt.plot(np.asarray(Ns), np.asarray(powers), linestyle='--', marker='o', color=colors[i], label=r'$\mathrm{cov}=$' + str(cov))

    plt.tight_layout()
    plt.legend(loc='upper left', fontsize=14)
    plt.xlabel('Number of genes, '+ r'$\mathrm{log}_{2}$', fontsize = 16)
    plt.xscale('log', basex=2)
    plt.axhline(0.05, color = 'dimgrey', lw = 2, ls = '--')
    plt.ylabel(r'$ \mathrm{P}\left ( \mathrm{reject} \; H_{0}   \mid H_{1} \;   \mathrm{is}\, \mathrm{true}, \, \alpha=0.05 \right ) $', fontsize = 16)
    fig_name = pt.get_path() + '/figs/poisson_power_G.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def poisson_neutral_fig(alpha = 0.05):
    df = pd.read_csv(pt.get_path() + '/data/simulations/ba_cov_neutral_sims.txt', sep='\t')
    neuts = np.sort(list(set(df.lambda_neutral.values)))
    cov = 0.2
    powers = []
    for neut in neuts:
        df_neut = df[ (df['lambda_neutral'] == neut)  ]
        p = df_neut.dist_percent.values
        p_sig = [p_i for p_i in p if p_i >= (1-alpha)]
        powers.append(len(p_sig) / len(p))
    fig = plt.figure()
    plt.plot(np.asarray(1 / neuts), np.asarray(powers), linestyle='--', marker='o', color='royalblue', label=r'$\mathrm{cov}=$' + str(cov))

    plt.tight_layout()
    plt.legend(loc='upper left', fontsize=14)
    plt.xscale('log', basex=10)

    plt.xlabel("Adaptive vs. non-adaptive substitution rate, " + r'$\frac{ \left \langle \lambda \right \rangle }{\lambda_{0}}$', fontsize = 16)
    plt.axhline(0.05, color = 'dimgrey', lw = 2, ls = '--')
    plt.ylabel(r'$ \mathrm{P}\left ( \mathrm{reject} \; H_{0}   \mid H_{1} \;   \mathrm{is}\, \mathrm{true}, \, \alpha=0.05 \right ) $', fontsize = 16)
    fig_name = pt.get_path() + '/figs/poisson_power_neutral.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()









#poisson_neutral_fig()
ltee_convergence()


#hist_tenaillon()
#tenaillon_p_N()
#poisson_power_G()
#poisson_power_N()
#tenaillon_p_N()
#def mean_euc_dist_fig():
#plot_permutation(dataset='good')

#fig1()
#tenaillon_fitness_hist()

#test_pca_regression()

#intro_fig()
#test_pca_regression()
#gene_space_fig()
#euc_dist_hist()
