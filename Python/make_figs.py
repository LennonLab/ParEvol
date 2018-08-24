from __future__ import division
import math, os, re
import numpy as np
import pandas as pd
import parevol_tools as pt
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table
from collections import Counter


def get_good_pca():
    df = pd.read_csv(pt.get_path() + '/data/Good_et_al/gene_by_pop_delta.txt', sep = '\t', header = 'infer', index_col = 0)

    to_exclude = pt.complete_nonmutator_lines()
    df = df[df.index.str.contains('|'.join( to_exclude))]

    X = pt.hellinger_transform(df)
    pca = PCA()
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(data=X_pca, index=df.index)
    #pt.plot_eigenvalues(pca.explained_variance_ratio_)

    times = sorted(list(set([int(x.split('_')[1]) for x in df.index.values])))
    colors = np.linspace(min(times),max(times),len(times))
    color_dict = dict(zip(times, colors))

    fig = plt.figure()
    plt.axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)
    plt.axvline(x=0, color='k', linestyle=':', alpha = 0.8, zorder=2)
    plt.scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=3)
    for pop in pt.complete_nonmutator_lines():
        pop_df_pca = df_pca[df_pca.index.str.contains(pop)]
        c_list = [ color_dict[int(x.split('_')[1])] for x in pop_df_pca.index.values]
        if  pt.nonmutator_shapes()[pop] == 'p2':
            size == 50
        else:
            size = 80
        plt.scatter(pop_df_pca.as_matrix()[:,0], pop_df_pca.as_matrix()[:,1], \
        c=c_list, cmap = cm.Blues, vmin=min(times), vmax=max(times), \
        marker = pt.nonmutator_shapes()[pop], s = size, edgecolors='#244162', linewidth = 0.6,  zorder=4)#, edgecolors='none')
    c = plt.colorbar()
    c.set_label("Generations", size=18)
    plt.xlim([-0.8,0.8])
    plt.ylim([-0.8,0.8])
    plt.xlabel('PCA 1 (' + str(round(pca.explained_variance_ratio_[0],3)*100) + '%)' , fontsize = 16)
    plt.ylabel('PCA 2 (' + str(round(pca.explained_variance_ratio_[1],3)*100) + '%)' , fontsize = 16)
    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/pca_good.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


    fig = plt.figure()
    for pop in pt.complete_nonmutator_lines():
        pop_df_pca = df_pca[df_pca.index.str.contains(pop)]
        x = [int(x.split('_')[1]) for x in pop_df_pca.index.values]
        c_list = [ color_dict[int(x.split('_')[1])] for x in pop_df_pca.index.values]
        if  pt.nonmutator_shapes()[pop] == 'p2':
            size == 50
        else:
            size = 80
        plt.scatter(x, pop_df_pca.as_matrix()[:,0], \
        c=c_list, cmap = cm.Blues, vmin=min(times), vmax=max(times), \
        marker = pt.nonmutator_shapes()[pop], s = size, edgecolors='#244162', \
        linewidth = 0.6, alpha = 0.7)#, edgecolors='none')
    plt.ylim([-0.7,0.7])
    plt.xlabel('Generations' , fontsize = 16)
    plt.ylabel('PCA 1 (' + str(round(pca.explained_variance_ratio_[0],3)*100) + '%)' , fontsize = 16)
    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/pca1_time_good.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


    fig = plt.figure()
    for pop in pt.complete_nonmutator_lines():
        pop_df_pca = df_pca[df_pca.index.str.contains(pop)]
        x = [int(x.split('_')[1]) for x in pop_df_pca.index.values]
        c_list = [ color_dict[int(x.split('_')[1])] for x in pop_df_pca.index.values]
        if  pt.nonmutator_shapes()[pop] == 'p2':
            size == 50
        else:
            size = 80
        plt.scatter(x, pop_df_pca.as_matrix()[:,1], \
        c=c_list, cmap = cm.Blues, vmin=min(times), vmax=max(times), \
        marker = pt.nonmutator_shapes()[pop], s = size, edgecolors='#244162', \
        linewidth = 0.6, alpha = 0.7)#, edgecolors='none')
    plt.ylim([-0.7,0.7])
    plt.xlabel('Generations' , fontsize = 16)
    plt.ylabel('PCA 2 (' + str(round(pca.explained_variance_ratio_[1],3)*100) + '%)' , fontsize = 16)
    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/pca2_time_good.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
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


def plot_permutation_sample_size():
    df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_delta = pt.likelihood_matrix(df, 'Tenaillon_et_al').get_likelihood_matrix()
    X = pt.hellinger_transform(df_delta)
    pca = PCA()
    df_out = pca.fit_transform(X)
    mcd = pt.get_mean_centroid_distance(df_out, k = 3)

    df_sample_path = pt.get_path() + '/data/Tenaillon_et_al/sample_size_permute_PCA.txt'
    df_sample = pd.read_csv(df_sample_path, sep = '\t', header = 'infer')#, index_col = 0)
    sample_sizes = sorted(list(set(df_sample.Sample_size.tolist())))

    fig = plt.figure()
    plt.axhline(mcd, color = 'k', lw = 3, ls = '--', zorder = 1)
    for sample_size in sample_sizes:
        df_sample_size = df_sample.loc[df_sample['Sample_size'] == sample_size]
        x_sample_size = df_sample_size.Sample_size.values
        y_sample_size = df_sample_size.MCD.values
        plt.scatter(x_sample_size, y_sample_size, c='#175ac6', marker = 'o', s = 70, \
            edgecolors='#244162', linewidth = 0.6, alpha = 0.3, zorder=2)#, edgecolors='none')

    plt.xlabel("Number of replicate populations", fontsize = 16)
    plt.ylabel("Mean centroid distance", fontsize = 16)

    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/plot_permutation_sample_size_tenaillon.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()






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
    plt.scatter(x, y, marker = "o", edgecolors='none', c = '#87CEEB', s = 120, zorder=3)
    #plt.plot(x, y)
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
    fittedvalues = data[:,2]
    pred_mean_se = data[:,3]
    pred_mean_ci_low, pred_mean_ci_upp = data[:,4:6].T
    pred_ci_low, pred_ci_upp = data[:,6:8].T


    slope_to_gamme = (1/slope) + 1

    plt.fill_between(x, pred_ci_low, pred_ci_upp, color='#175ac6', lw=0.5, alpha=0.2)
    #'$^\frac{1}{1 - '+str(round(slope_to_gamme,2))+'}$'
    #plt.text(2.4, 2.1, r'$k_{max}$'+ ' = '+str(round(10**intercept,2))+'*'+r'$t$'+ '$^{\frac{1}{1 - '+str(round(slope_to_gamme,2))+'}}$', fontsize=10, color='k', alpha=0.9)
    plt.text(2.4, 2.05, r'$k_{max}$'+ ' = ' + str(round(10**intercept,2)) + '*' + r'$t^ \frac{1}{1 - ' + str(round(slope_to_gamme,2)) + '}$', fontsize=12, color='k', alpha=0.9)
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






#plot_pcoa('tenaillon')
#example_gene_space()
#plot_permutation('good')
#plot_network()
#plot_kmax_over_time()
plot_permutation_sample_size()
