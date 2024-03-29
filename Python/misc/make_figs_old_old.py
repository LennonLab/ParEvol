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

mydir = os.path.expanduser("~/GitHub/ParEvol/")


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
    ax3.set_ylabel(r'$  \frac{\left | \mathcal{G}_{1}\left ( k \right )  \right | }{\left | \mathcal{G}_{0}\left ( k \right )  \right | }$', fontsize = 15)

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
    ax4.set_ylabel(r'$ \frac{\left | \mathcal{G}_{1}\left ( k \right )  \right | }{\left | \mathcal{G}_{0}\left ( k \right )  \right | }$', fontsize = 15)
    ax4.legend(loc='lower left', fontsize=7)
    ax4.set_yscale("log")
    plt.tight_layout()
    fig_name = mydir + 'figs/test_network.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()













def tenaillon_p_N(alpha = 0.05, bootstraps=10000, bootstrap_sample = 100):
    fig = plt.figure()
    df = pd.read_csv(mydir + 'data/Tenaillon_et_al/dist_sample_size.txt', sep='\t')
    Ns = list(set(df.N.values))
    for N in Ns:
        print(N)
        N_dist = df.loc[df['N'] == N].dist_percent.values
        bootstrap_power = []
        for bootstrap in range(bootstraps):
            p_sig = [p_i for p_i in np.random.choice(N_dist, size = bootstrap_sample) if p_i < alpha]
            bootstrap_power.append(len(p_sig) / bootstrap_sample)
        bootstrap_power = np.sort(bootstrap_power)
        N_power = len([p_i for p_i in N_dist if p_i <  alpha]) / len(N_dist)
        lower_ci = bootstrap_power[int(len(bootstrap_power) * 0.05)]
        upper_ci = bootstrap_power[  len(bootstrap_power) -  int(len(bootstrap_power) * 0.05)]
        plt.errorbar(N, N_power, yerr = [np.asarray([N_power-upper_ci]), np.asarray([lower_ci - N_power])], fmt = 'o', alpha = 0.5, \
            barsabove = True, marker = '.', mfc = 'k', mec = 'k', c = 'k', zorder=1)
        plt.scatter(N, N_power, c='#175ac6', marker = 'o', s = 70, \
            edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)
    plt.xlabel('Number of replicate populations', fontsize = 16)
    plt.ylabel('Bootstrapped statistical power', fontsize = 16)
    plt.axhline(0.05, color = 'dimgrey', lw = 2, ls = '--')
    plt.tight_layout()
    fig_name = mydir + 'figs/tenaillon_N.png'
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





def wannier_hist(iter=10000):
    dir = os.path.expanduser("~/GitHub/ParEvol")
    df1 = pd.read_csv(dir + '/data/Wannier_et_al/C321.deltaA_mutation_table_clean.txt', sep='\t', index_col=0)
    df2 = pd.read_csv(dir + '/data/Wannier_et_al/C321_mutation_table_clean.txt', sep='\t', index_col=0)
    df = df1.append(df2, sort=False)
    df = df.fillna(0)
    df_np = df.values
    gene_names = df.columns.values
    N1 = df1.shape[0]
    N2 = df2.shape[0]
    df_np_delta = cd.likelihood_matrix_array(df_np, gene_names, 'Wannier_et_al').get_likelihood_matrix()
    F2_all = pt.get_F_2(df_np_delta, N1, N2)
    print(F2_all)
    F2 = F2_all[0]
    V1 = F2_all[1]
    V2 = F2_all[2]

    F2_null = []
    V1_null = []
    V2_null = []
    for i in range(iter):
        if i %1000 ==0:
            print(i)
        df_np_i = pt.get_random_matrix(df_np)
        np.seterr(divide='ignore')
        df_np_i_delta = cd.likelihood_matrix_array(df_np_i, gene_names, 'Wannier_et_al').get_likelihood_matrix()
        F2_all_iter = pt.get_F_2(df_np_i_delta, N1, N2)
        F2_null.append(F2_all_iter[0])
        V1_null.append(F2_all_iter[1])
        V2_null.append(F2_all_iter[2])

    fig = plt.figure()
    #plt.hist(F2_null, bins=30, weights=np.zeros_like(F2_null) + 1. / len(F2_null), alpha=0.8, color = '#175ac6')
    plt.hist(F2_null, bins=30, alpha=0.8, color = '#175ac6')
    plt.axvline(F2, color = 'red', lw = 3)
    plt.xlabel(r'$ F_{2}$', fontsize = 20)
    plt.ylabel("Frequency", fontsize = 12)
    fig.tight_layout()
    fig.savefig(os.path.expanduser("~/GitHub/ParEvol") + '/figs/test_hist_F.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


    fig = plt.figure()
    plt.hist(V1_null, bins=30, alpha=0.8, color = '#175ac6')
    plt.axvline(V1, color = 'red', lw = 3)
    plt.xlabel(r'$ V_{1}$', fontsize = 20)
    plt.ylabel("Frequency", fontsize = 12)
    fig.tight_layout()
    fig.savefig(os.path.expanduser("~/GitHub/ParEvol") + '/figs/test_hist_F_V1.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()

    fig = plt.figure()
    plt.hist(V2_null, bins=30, alpha=0.8, color = '#175ac6')
    #print(V2_null)
    plt.axvline(V2, color = 'red', lw = 3)
    plt.xlabel(r'$ V_{2}$', fontsize = 20)
    plt.ylabel("Frequency", fontsize = 12)
    fig.tight_layout()
    fig.savefig(os.path.expanduser("~/GitHub/ParEvol") + '/figs/test_hist_F_V2.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()








#hist_tenaillon_multi_pop()

#spacing_dist()

#wannier_hist()

#poisson_neutral_fig()
#ltee_convergence()

#intro_fig()
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
#power_figs()
#intro_fig()
#test_pca_regression()
#gene_space_fig()
#euc_dist_hist()
