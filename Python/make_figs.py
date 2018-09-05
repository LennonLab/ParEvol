from __future__ import division
import math, os, re
import numpy as np
import pandas as pd
import parevol_tools as pt
import matplotlib.pyplot as plt
from matplotlib import cm, rc_context
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table
from collections import Counter

def fig1(k = 3):
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
    mean_length = pt.get_euclidean_distance(df_out, k=k)

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
    delta_L_list = df_null.delta_L.tolist()
    #ax3.hist(delta_L_list, bins=30, histtype='stepfilled', normed=True, alpha=0.6, color='b')
    ax3.hist(delta_L_list,bins=30, weights=np.zeros_like(delta_L_list) + 1. / len(delta_L_list), alpha=0.8, color = '#175ac6')
    ax3.axvline(mean_length, color = 'red', lw = 3)
    ax3.set_xlabel("Mean pairwise difference \n in magnitudes, " + r'$   \left \langle  \left | \Delta L \right |\right \rangle$', fontsize = 14)
    ax3.set_ylabel("Frequency", fontsize = 16)

    delta_L_list.append(mean_length)
    relative_position_delta_L = sorted(delta_L_list).index(mean_length) / (len(delta_L_list) -1)
    if relative_position_delta_L > 0.5:
        p_score_delta_L = 1 - relative_position_delta_L
    else:
        p_score_delta_L = relative_position_delta_L
    print('mean difference in magnitudes p-score = ' + str(round(p_score_delta_L, 3)))
    ax3.text(0.078, 0.115, r'$p < 0.05$', fontsize = 10)

    ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax4_values = df_null.mean_angle.values
    ax4_values = ax4_values[np.logical_not(np.isnan(ax4_values))]
    #ax4.hist(ax4_values, bins=30, histtype='stepfilled', normed=True, alpha=0.6, color='b')
    ax4.hist(ax4_values,bins=30, weights=np.zeros_like(ax4_values) + 1. / len(ax4_values), alpha=0.8, color = '#175ac6')
    ax4.axvline(mean_angle, color = 'red', lw = 3)
    ax4.set_xlabel("Mean pairwise angle, " + r'$\left \langle \theta \right \rangle$', fontsize = 14)
    ax4.set_ylabel("Frequency", fontsize = 16)

    mean_angle_list = ax4_values.tolist()
    mean_angle_list.append(mean_angle)
    relative_position_angle = sorted(mean_angle_list).index(mean_angle) / (len(mean_angle_list) -1)
    if relative_position_angle > 0.5:
        p_score_angle = 1 - relative_position_angle
    else:
        p_score_angle = relative_position_angle
    print('mean pairwise angle p-score = ' + str(round(p_score_angle, 3)))
    ax4.text(89.1, 0.09, r'$p \nless  0.05$', fontsize = 10)

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


#def fig3():
#


def fig4():
    df_path = pt.get_path() + '/data/Tenaillon_et_al/network.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    dist_df = pt.networkx_distance(df)
    df_C_path = pt.get_path() + '/data/Tenaillon_et_al/network_CCs.txt'
    df_C = pd.read_csv(df_C_path, sep = '\t', header = 'infer', index_col = 0)
    kmax_df = max(df_C.k_i.values)
    mean_C_df = np.mean(df_C.loc[df_C['k_i'] >= 2].C_i.values)
    df_null_path = pt.get_path() + '/data/Tenaillon_et_al/permute_network.txt'
    df_null = pd.read_csv(df_null_path, sep = '\t', header = 'infer', index_col = 0)

    C_mean_null = df_null.C_mean_no1or2.tolist()
    C_mean_null = [x for x in C_mean_null if str(x) != 'nan']
    d_mean_null = df_null.d_mean.tolist()
    k_max_null = df_null.k_max.tolist()

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    # this is a  placeholder. I need to figure out how to plot a network.....
    ax1.scatter(k_max_null, d_mean_null, marker = "o", edgecolors='#244162', c = '#175ac6', alpha = 0.4, s = 60, zorder=4)
    #ax1.set_xlim([-0.75,0.75])
    #ax1.set_ylim([-0.75,0.75])


    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    ax2.hist(k_max_null,bins=30, weights=np.zeros_like(k_max_null) + 1. / len(k_max_null), alpha=0.8, color = '#175ac6')
    ax2.axvline(kmax_df, color = 'red', lw = 3)
    ax2.set_xlabel("kmax", fontsize = 14)
    ax2.set_ylabel("Frequency", fontsize = 16)

    k_max_null.append(kmax_df)
    relative_position_k_max = sorted(k_max_null).index(kmax_df) / (len(k_max_null) -1)
    if relative_position_k_max > 0.5:
        p_score_k_max = 1 - relative_position_k_max
    else:
        p_score_k_max = relative_position_k_max
    print('kmax p-score = ' + str(round(p_score_k_max, 3)))
    #ax2.text(0.366, 0.088, r'$p < 0.05$', fontsize = 10)

    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    #print(C_mean_null)
    ax3.hist(C_mean_null,bins=30, weights=np.zeros_like(C_mean_null) + 1. / len(C_mean_null), alpha=0.8, color = '#175ac6')
    ax3.axvline(mean_C_df, color = 'red', lw = 3)
    ax3.set_xlabel("Mean clustering coefficient", fontsize = 14)
    ax3.set_ylabel("Frequency", fontsize = 16)

    C_mean_null.append(mean_C_df)
    relative_position_mean_C = sorted(C_mean_null).index(mean_C_df) / (len(C_mean_null) -1)
    if relative_position_mean_C > 0.5:
        p_score_mean_C = 1 - relative_position_mean_C
    else:
        p_score_mean_C = relative_position_mean_C
    print('mean C p-score = ' + str(round(p_score_mean_C, 3)))
    #ax3.text(0.078, 0.115, r'$p < 0.05$', fontsize = 10)


    ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax4.hist(d_mean_null,bins=30, weights=np.zeros_like(d_mean_null) + 1. / len(d_mean_null), alpha=0.8, color = '#175ac6')
    ax4.axvline(dist_df, color = 'red', lw = 3)
    ax4.set_xlabel("Mean distance", fontsize = 14)
    ax4.set_ylabel("Frequency", fontsize = 16)

    d_mean_null.append(dist_df)
    relative_position_d_mean = sorted(d_mean_null).index(dist_df) / (len(d_mean_null) -1)
    if relative_position_d_mean > 0.5:
        p_score_d_mean = 1 - relative_position_d_mean
    else:
        p_score_d_mean = relative_position_d_mean
    print('mean pairwise angle p-score = ' + str(round(p_score_d_mean, 3)))
    #ax4.text(89.1, 0.09, r'$p \nless  0.05$', fontsize = 10)

    plt.tight_layout()
    fig_name = pt.get_path() + '/figs/fig4.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



# def fig5():
# effect of sample size on featyres




def fig6():
    network_dir = pt.get_path() + '/data/Good_et_al/networks_BIC/'
    time_nodes = []
    time_kmax = []
    for filename in os.listdir(network_dir):
        df = pd.read_csv(network_dir + filename, sep = '\t', header = 'infer', index_col = 0)
        gens = filename.split('.')
        time = re.split('[_.]', filename)[1]
        time_nodes.append((int(time), df.shape[0]))
        time_kmax.append((int(time), max(df.astype(bool).sum(axis=0).values)))

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    time_nodes_sorted = sorted(time_nodes, key=lambda tup: tup[0])
    x_nodes = [i[0] for i in time_nodes_sorted]
    y_nodes = [i[1] for i in time_nodes_sorted]
    ax1.scatter(x_nodes, y_nodes, marker = "o", edgecolors='#244162', \
        c = '#175ac6', s = 80, zorder=3, alpha = 0.6)
    ax1.set_xlabel("Time (generations)", fontsize = 14)
    ax1.set_ylabel('Network size, ' + r'$N$', fontsize = 14)
    ax1.set_ylim(5, 500)
    ax1.set_yscale('log')


    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    time_kmax_sorted = sorted(time_kmax, key=lambda tup: tup[0])
    x_kmax = [i[0] for i in time_kmax_sorted]
    y_kmax = [i[1] for i in time_kmax_sorted]
    x_kmax = np.log10(x_kmax)
    y_kmax = np.log10(y_kmax)
    ax2.scatter(x_kmax, y_kmax, c='#175ac6', marker = 'o', s = 80, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.6, zorder=1)#, edgecolors='none')

    '''The below regression code is from the GitHub repository
    ScalingMicroBiodiversity and is licensed under a
    GNU General Public License v3.0.

    https://github.com/klocey/ScalingMicroBiodiversity
    '''
    df_regression = pd.DataFrame({'t': list(x_kmax)})
    df_regression['kmax'] = list(y_kmax)
    f = smf.ols('kmax ~ t', df_regression).fit()

    R2 = f.rsquared
    pval = f.pvalues
    intercept = f.params[0]
    slope = f.params[1]
    X = np.linspace(min(x_kmax), max(x_kmax), 1000)
    Y = f.predict(exog=dict(t=X))
    print(min(x_kmax), max(y_kmax))

    st, data, ss2 = summary_table(f, alpha=0.05)
    fittedvalues = data[:,2]
    pred_mean_se = data[:,3]
    pred_mean_ci_low, pred_mean_ci_upp = data[:,4:6].T
    pred_ci_low, pred_ci_upp = data[:,6:8].T

    slope_to_gamme = (1/slope) + 1

    ax2.fill_between(x_kmax, pred_ci_low, pred_ci_upp, color='#175ac6', lw=0.5, alpha=0.2, zorder=2)
    ax2.text(2.35, 2.03, r'$k_{max}$'+ ' = ' + str(round(10**intercept,2)) + '*' + r'$t^ \frac{1}{\,' + str(round(slope_to_gamme,2)) + '- 1}$', fontsize=9, color='k', alpha=0.9)
    ax2.text(2.35, 1.83,  r'$r^2$' + ' = ' +str("%.2f" % R2), fontsize=9, color='0.2')
    ax2.plot(X.tolist(), Y.tolist(), '--', c='k', lw=2, alpha=0.8, color='k', label='Power-law', zorder=2)
    ax2.set_xlabel("Time (generations), " + r'$\mathrm{log}_{10}$', fontsize = 14)
    ax2.set_ylabel(r'$k_{max}, \;  \mathrm{log}_{10}$', fontsize = 14)


    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1)

    df_net_feats_path = pt.get_path() + '/data/Good_et_al/network_features.txt'
    df_net_feats = pd.read_csv(df_net_feats_path, sep = '\t', header = 'infer')
    x_C = df_net_feats.N.values
    y_C = df_net_feats.C_mean.values

    ax3.scatter(x_C, y_C, c='#175ac6', marker = 'o', s = 80, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.6, zorder=1)
    x_C_range = list(range(10, max(x_C)))
    barabasi_albert_range_C = [  ((np.log(i) ** 2) / i) for i in x_C_range ]
    random_range_c = [ (1/i) for i in x_C_range ]

    x_C_sort = list(set(x_C.tolist()))
    x_C_sort.sort()
    model = pt.clusterBarabasiAlbert(x_C, y_C)
    b0_start = [0.01, 0.1, 1, 10]
    z_start = [-2,-0.5]
    results = []
    for b0 in b0_start:
        for z in z_start:
            start_params = [b0, z]
            result = model.fit(start_params = start_params)
            results.append(result)
    AICs = [result.aic for result in results]
    best = results[AICs.index(min(AICs))]
    best_CI_FIC = pt.CI_FIC(best)
    best_CI = best.conf_int()
    best_params = best.params

    barabasi_albert_range_C_ll = pt.cluster_BA(np.sort(x_C), best_params[0])

    ax3.plot(np.sort(x_C), barabasi_albert_range_C_ll, c = 'k', lw = 2.5,
        ls = '--', zorder=2)
    #plt.plot(x_C_range, random_range_c, c = 'r', lw = 2.5, ls = '--')
    ax3.set_xlabel('Network size, ' + r'$N$', fontsize = 14)
    ax3.set_ylabel('Mean clustering \ncoefficient, ' + r'$\left \langle C \right \rangle$', fontsize = 14)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylim(0.05, 1.5)


    ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    x_d = df_net_feats.N.values
    y_d = df_net_feats.d_mean.values
    ax4.scatter(x_d, y_d, c='#175ac6', marker = 'o', s = 80, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.6, zorder=1)
    #x_d_range = list(range(10, max(x_d)))
    #barabasi_albert_range_d = [ (np.log(i) / np.log(np.log(i))) for i in x_d_range ]
    x_d_sort = list(set(x_d.tolist()))
    x_d_sort.sort()
    model_d = pt.distanceBarabasiAlbert(x_d, y_d)
    results_d = []
    for b0 in b0_start:
        for z in z_start:
            start_params_d = [b0, z]
            result_d = model_d.fit(start_params = start_params_d)
            results_d.append(result_d)
    AICs_d = [result_d.aic for result_d in results_d]
    best_d = results_d[AICs_d.index(min(AICs_d))]
    best_CI_FIC_d = pt.CI_FIC(best_d)
    best_CI_d = best_d.conf_int()
    best_d_params = best_d.params



    barabasi_albert_range_d_ll = pt.distance_BA(np.sort(x_d), best_d_params[0])


    ax4.plot(np.sort(x_C), barabasi_albert_range_d_ll, c = 'k', lw = 2.5,
        ls = '--', zorder = 2)
    #random_range = [ np.log(i) for i in x_d_range ]
    #ax4.plot(x_d_range, random_range, c = 'r', lw = 2.5, ls = '--')
    ax4.set_xlabel('Network size, ' + r'$N$', fontsize = 14)
    ax4.set_ylabel('Mean distance, ' + r'$\left \langle d \right \rangle$', fontsize = 14)

    plt.tight_layout()
    fig_name = pt.get_path() + '/figs/fig6.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()









def plot_permutation(dataset, analysis = 'PCA', alpha = 0.05):
    if dataset == 'tenaillon':
        df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        df_delta = pt.likelihood_matrix(df, 'Tenaillon_et_al').get_likelihood_matrix()
        if analysis == 'PCA':
            X = pt.hellinger_transform(df_delta)
            pca = PCA()
            df_out = pca.fit_transform(X)
        elif analysis == 'cMDS':
            df_delta_bc = np.sqrt(pt.get_scipy_bray_curtis(df_delta.as_matrix()))
            df_out = pt.cmdscale(df_delta_bc)[0]

        mcd = pt.get_mean_centroid_distance(df_out, k = 3)

        mcd_perm_path = pt.get_path() + '/data/Tenaillon_et_al/permute_' + analysis + '.txt'
        mcd_perm = pd.read_csv(mcd_perm_path, sep = '\t', header = 'infer', index_col = 0)
        mcd_perm_list = mcd_perm.MCD.tolist()
        iterations = len(mcd_perm_list)
        mcd_perm_list.append(mcd)
        relative_position = sorted(mcd_perm_list).index(mcd) / iterations
        if relative_position > 0.5:
            p_score = 1 - (sorted(mcd_perm_list).index(mcd) / iterations)
        else:
            p_score = (sorted(mcd_perm_list).index(mcd) / iterations)
        print(p_score)

        fig = plt.figure()
        plt.hist(mcd_perm_list, bins=30, histtype='stepfilled', normed=True, alpha=0.6, color='b')
        plt.axvline(mcd, color = 'red', lw = 3)
        plt.xlabel("Mean centroid distance", fontsize = 18)
        plt.ylabel("Frequency", fontsize = 18)
        fig.tight_layout()
        plot_out = pt.get_path() + '/figs/permutation_hist_tenaillon_' + analysis + '.png'
        fig.savefig(plot_out, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
        plt.close()

    elif dataset == 'good':
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
            mcds.append(pt.get_mean_centroid_distance(df_rndm_delta_out_tp.as_matrix(), k=3))

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
            mcd_perm_y_sort = np.sort(mcd_perm_y.MCD.tolist())
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
        plt.ylabel("Mean \n centroid distance", fontsize = 14)

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
        plt.ylabel("Standardized mean \n centroid distance", fontsize = 14)

        fig.tight_layout()
        fig.savefig(pt.get_path() + '/figs/permutation_scatter_good.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
        plt.close()

    else:
        print('Dataset argument not accepted')






####### network figs



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


def plot_cluster_dist():
    df_path = pt.get_path() + '/data/Tenaillon_et_al/network_CCs.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    k_count = dict(Counter(df.C_i.values))
    k_count = {k: v / total for total in (sum(k_count.values()),) for k, v in k_count.items()}
    #x = np.log10(list(k_count.keys()))
    #y = np.log10(list(k_count.values()))
    # cluster kde
    C_i = df.C_i.values
    grid_ = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.1, 10, 50)},
                    cv=20) # 20-fold cross-validation
    grid_.fit(C_i[:, None])
    x_grid_ = np.linspace(0, 2.5, 1000)
    kde_ = grid_.best_estimator_
    pdf_ = np.exp(kde_.score_samples(x_grid_[:, None]))
    pdf_ = [x / sum(pdf_) for x in pdf_]

    x = list(k_count.keys())
    y = list(k_count.values())

    #x_poisson = list(range(1, 100))
    #y_poisson = [(math.exp(-k_mean) * ( (k_mean ** k)  /  math.factorial(k) )) for k in x_poisson]

    fig = plt.figure()
    #plt.scatter(x, y, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=3)
    plt.plot(x_grid_, pdf_)
    plt.ylabel("Clustering coefficient, " + r'$C_{i}$', fontsize = 16)
    plt.xlabel("Number of edges, " + r'$k$', fontsize = 16)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim(0, 1)
    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/C_dist.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def plot_C_vs_k_tenaillon():
    df_path = pt.get_path() + '/data/Tenaillon_et_al/network_CCs.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    fig = plt.figure()
    plt.scatter(df.k_i.values, df.C_i.values, marker = "o", edgecolors='#244162', c = '#175ac6', s = 120, zorder=3)
    plt.ylabel("Clustering coefficient, " + r'$C_{k}$', fontsize = 16)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlim(0, 20)
    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/C_vs_k_tenaillon.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_nodes_over_time():
    directory = pt.get_path() + '/data/Good_et_al/networks_BIC/'
    time_nodes = []
    for filename in os.listdir(directory):
        df = pd.read_csv(directory + filename, sep = '\t', header = 'infer', index_col = 0)
        gens = filename.split('.')
        time = re.split('[_.]', filename)[1]
        time_nodes.append((int(time), df.shape[0]))
    time_nodes_sorted = sorted(time_nodes, key=lambda tup: tup[0])
    x = [i[0] for i in time_nodes_sorted]
    y = [i[1] for i in time_nodes_sorted]

    x_pred = list(set(x))
    x_pred.sort()
    y_pred = [min(y) + x_pred_i + 1 for x_pred_i in list(range(len(x_pred) ))]


    fig = plt.figure()
    plt.scatter(x, y, marker = "o", edgecolors='#244162', c = '#175ac6', s = 120, zorder=3)
    plt.plot(x_pred, y_pred)
    plt.xlabel("Time (generations)", fontsize = 18)
    plt.ylabel('Network size, ' + r'$N$', fontsize = 18)
    plt.ylim(5, 500)
    plt.yscale('log')

    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/good_N_vs_time.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
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

    #df_rndm_path = pt.get_path() + '/data/Good_et_al/networks_BIC_rndm.txt'
    #df_rndm = pd.read_csv(df_rndm_path, sep = '\t', header = 'infer')

    #x_rndm = np.log10(df_rndm.Generations.values)
    #y_rndm = np.log10(df_rndm.Generations.values)

    fig = plt.figure()
    #plt.scatter(x, y, marker = "o", edgecolors='none', c = '#175ac6', s = 120, zorder=3)
    plt.scatter(x, y, c='#175ac6', marker = 'o', s = 120, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.8, zorder=3)#, edgecolors='none')
    #plt.scatter(x_rndm, y_rndm, marker = "o", edgecolors='none', c = 'blue', s = 120, alpha = 0.1)
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
    print(ss2)
    fittedvalues = data[:,2]
    pred_mean_se = data[:,3]
    pred_mean_ci_low, pred_mean_ci_upp = data[:,4:6].T
    pred_ci_low, pred_ci_upp = data[:,6:8].T

    slope_to_gamme = (1/slope) + 1

    plt.fill_between(x, pred_ci_low, pred_ci_upp, color='#175ac6', lw=0.5, alpha=0.2)
    #'$^\frac{1}{1 - '+str(round(slope_to_gamme,2))+'}$'
    #plt.text(2.4, 2.1, r'$k_{max}$'+ ' = '+str(round(10**intercept,2))+'*'+r'$t$'+ '$^{\frac{1}{1 - '+str(round(slope_to_gamme,2))+'}}$', fontsize=10, color='k', alpha=0.9)
    plt.text(2.4, 2.05, r'$k_{max}$'+ ' = ' + str(round(10**intercept,2)) + '*' + r'$t^ \frac{1}{\,' + str(round(slope_to_gamme,2)) + '- 1}$', fontsize=12, color='k', alpha=0.9)
    plt.text(2.4, 1.94,  r'$r^2$' + ' = ' +str("%.2f" % R2), fontsize=12, color='0.2')
    plt.plot(X.tolist(), Y.tolist(), '--', c='k', lw=2, alpha=0.8, color='k', label='Power-law')

    #plt.plot(t_x, t_y)
    plt.xlabel("Time (generations), " + r'$\mathrm{log}_{10}$', fontsize = 18)
    plt.ylabel(r'$k_{max}, \;  \mathrm{log}_{10}$', fontsize = 18)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim(0.001, 1)

    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/good_kmax_vs_time.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_cluster():
    df_path = pt.get_path() + '/data/Good_et_al/network_features.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer')

    fig = plt.figure()
    x = df.N.values
    y = df.C_mean.values
    #plt.scatter(x, y, marker = "o", edgecolors='none', c = '#87CEEB', s = 120, zorder=3)
    plt.scatter(x, y, c='#175ac6', marker = 'o', s = 120, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.9, zorder=3)

    x_range = list(range(10, max(x)))
    barabasi_albert_range = [ ((np.log(i) ** 2) / i) for i in x_range ]
    random_range = [(1/i) for i in x_range ]
    plt.plot(x_range, barabasi_albert_range, c = 'k', lw = 2.5, ls = '--')

    plt.xlabel('Network size, ' + r'$N$', fontsize = 18)
    plt.ylabel('Mean clustering coefficient, ' + r'$\left \langle C \right \rangle$', fontsize = 16)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(0.05, 1.5)

    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/good_N_vs_C.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def plot_distance():
    df_path = pt.get_path() + '/data/Good_et_al/network_features.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer')

    fig = plt.figure()
    x = df.N.values
    y = df.d_mean.values
    #plt.scatter(x, y, marker = "o", edgecolors='none', c = '#87CEEB', s = 120, zorder=3)
    plt.scatter(x, y, c='#175ac6', marker = 'o', s = 120, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.9, zorder=3)

    x_range = list(range(10, max(x)))
    barabasi_albert_range = [ (np.log(i) / np.log(np.log(i))) for i in x_range ]
    random_range = [np.log(i) for i in x_range ]
    plt.plot(x_range, barabasi_albert_range, c = 'r', lw = 2.5, ls = '--')
    plt.plot(x_range, random_range, c = 'k', lw = 2.5, ls = '--')

    plt.xlabel('Network size, ' + r'$N$', fontsize = 18)
    plt.ylabel('Mean distance, ' + r'$\left \langle d \right \rangle$', fontsize = 16)
    #plt.xscale('log')
    #plt.ylim(0.05, 1.5)

    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/good_N_vs_d.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()





#plot_pcoa('tenaillon')
#example_gene_space()
#plot_permutation('good')
#plot_network()
#plot_kmax_over_time()
#plot_permutation_sample_size()
#plot_distance()
#plot_cluster_dist()
#get_tenaillon_pca()
#plot_nodes_over_time()

#plot_C_vs_k_tenaillon()

#fig2()

#plot_kmax_over_time()
#fig4()
#plot_nodes_over_time()

fig6()
