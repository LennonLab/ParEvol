from __future__ import division
import math
import numpy as np
import pandas as pd
import parevol_tools as pt
import matplotlib.pyplot as plt
import matplotlib.colors as cls
from matplotlib import cm
from sklearn.decomposition import PCA


def plot_pcoa(dataset):
    if dataset == 'tenaillon':
        df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop_delta.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        # square root transformation to remove negative eigenvalues
        X = np.sqrt(pt.get_bray_curtis(df.as_matrix()))
        X_cmds = pt.cmdscale(X)
        #pt.plot_eigenvalues(X_cmds[1], file_name = 'pcoa_tenaillon_eigen')
        percent_explained = X_cmds[1] / sum(X_cmds[1])
        names = df.index.values
        fig = plt.figure()
        plt.axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)
        plt.axvline(x=0, color='k', linestyle=':', alpha = 0.8, zorder=2)
        plt.scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=3)
        plt.scatter(X_cmds[0][:,0], X_cmds[0][:,1], marker = "o", edgecolors='none', c = 'forestgreen', alpha = 0.6, s = 80, zorder=4)

        plt.xlim([-0.4,0.4])
        plt.ylim([-0.4,0.4])
        plt.xlabel('PCoA 1 (' + str(round(percent_explained[0],3)*100) + '%)' , fontsize = 16)
        plt.ylabel('PCoA 2 (' + str(round(percent_explained[1],3)*100) + '%)' , fontsize = 16)
        fig.tight_layout()
        fig.savefig(pt.get_path() + '/figs/pcoa_tenaillon.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
        plt.close()

    elif dataset == 'good':
        df_path = pt.get_path() + '/data/Good_et_al/gene_by_pop_delta.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        to_exclude = pt.complete_nonmutator_lines()
        df = df[df.index.str.contains('|'.join( to_exclude))]
        # square root transformation to remove negative eigenvalues
        X = np.sqrt(pt.get_scipy_bray_curtis(df.as_matrix()))
        #X = pt.get_scipy_bray_curtis(df.as_matrix())
        X_cmds = pt.cmdscale(X)
        df_cmds = pd.DataFrame(data=X_cmds[0], index=df.index)
        #pt.plot_eigenvalues(X_cmds[1], file_name = 'pcoa_good_eigen')
        percent_explained = X_cmds[1] / sum(X_cmds[1])
        times = sorted(list(set([int(x.split('_')[1]) for x in df.index.values])))
        colors = np.linspace(min(times),max(times),len(times))
        color_dict = dict(zip(times, colors))

        fig = plt.figure()
        plt.axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)
        plt.axvline(x=0, color='k', linestyle=':', alpha = 0.8, zorder=2)
        plt.scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=1)
        for pop in pt.complete_nonmutator_lines():
            pop_df_cmds = df_cmds[df_cmds.index.str.contains(pop)]
            c_list = [ color_dict[int(x.split('_')[1])] for x in pop_df_cmds.index.values]
            if  pt.nonmutator_shapes()[pop] == 'p2':
                size == 50
            else:
                size = 80
            plt.scatter(pop_df_cmds.as_matrix()[:,0], pop_df_cmds.as_matrix()[:,1], \
            c=c_list, cmap = cm.Blues, vmin=min(times), vmax=max(times), \
            marker = pt.nonmutator_shapes()[pop], s = size, edgecolors='#244162', linewidth = 0.6)#, edgecolors='none')
        c = plt.colorbar()
        c.set_label("Generations")
        plt.xlim([-0.7,0.7])
        plt.ylim([-0.7,0.7])
        plt.xlabel('PCoA 1 (' + str(round(percent_explained[0],3)*100) + '%)' , fontsize = 16)
        plt.ylabel('PCoA 2 (' + str(round(percent_explained[1],3)*100) + '%)' , fontsize = 16)
        fig.tight_layout()
        fig.savefig(pt.get_path() + '/figs/pcoa_good.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
        plt.close()


        fig = plt.figure()
        for pop in pt.complete_nonmutator_lines():
            pop_df_cmds = df_cmds[df_cmds.index.str.contains(pop)]
            x = [int(x.split('_')[1]) for x in pop_df_cmds.index.values]
            c_list = [ color_dict[int(x.split('_')[1])] for x in pop_df_cmds.index.values]
            if  pt.nonmutator_shapes()[pop] == 'p2':
                size == 50
            else:
                size = 80
            plt.scatter(x, pop_df_cmds.as_matrix()[:,0], \
            c=c_list, cmap = cm.Blues, vmin=min(times), vmax=max(times), \
            marker = pt.nonmutator_shapes()[pop], s = size, edgecolors='#244162', \
            linewidth = 0.6, alpha = 0.7)#, edgecolors='none')
        plt.ylim([-0.7,0.7])
        plt.xlabel('Generations' , fontsize = 16)
        plt.ylabel('PCoA 1 (' + str(round(percent_explained[0],3)*100) + '%)' , fontsize = 16)
        fig.tight_layout()
        fig.savefig(pt.get_path() + '/figs/pcoa1_time_good.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
        plt.close()


        fig = plt.figure()
        for pop in pt.complete_nonmutator_lines():
            pop_df_cmds = df_cmds[df_cmds.index.str.contains(pop)]
            x = [int(x.split('_')[1]) for x in pop_df_cmds.index.values]
            c_list = [ color_dict[int(x.split('_')[1])] for x in pop_df_cmds.index.values]
            if  pt.nonmutator_shapes()[pop] == 'p2':
                size == 50
            else:
                size = 80
            plt.scatter(x, pop_df_cmds.as_matrix()[:,1], \
            c=c_list, cmap = cm.Blues, vmin=min(times), vmax=max(times), \
            marker = pt.nonmutator_shapes()[pop], s = size, edgecolors='#244162', \
            linewidth = 0.6, alpha = 0.7)#, edgecolors='none')
        plt.ylim([-0.7,0.7])
        plt.xlabel('Generations' , fontsize = 16)
        plt.ylabel('PCoA 2 (' + str(round(percent_explained[1],3)*100) + '%)' , fontsize = 16)
        fig.tight_layout()
        fig.savefig(pt.get_path() + '/figs/pcoa2_time_good.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
        plt.close()



def plot_permutation(dataset, analysis = 'PCA'):
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
        plt.axvline(mcd, color = 'red')
        plt.xlabel("Mean centroid distance")
        plt.ylabel("Frequency")
        fig.tight_layout()
        plot_out = pt.get_path() + '/figs/permutation_hist_tenaillon_' + analysis + '.png'
        fig.savefig(plot_out, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
        plt.close()

    else:
        print('finish this function')


def plot_mcd_pcoa_good():
    df_path = pt.get_path() + '/data/Good_et_al/gene_by_pop_delta.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_null_path = pt.get_path() + '/data/Good_et_al/permute.txt'
    df_null = pd.read_csv(df_null_path, sep = '\t', header = 'infer', index_col = 0)
    #print(df_null)
    to_exclude = pt.complete_nonmutator_lines()
    to_exclude.append('p5')
    df = df[df.index.str.contains('|'.join( to_exclude))]
    # square root transformation to remove negative eigenvalues
    X = np.sqrt(pt.get_scipy_bray_curtis(df.as_matrix()))
    X_cmds = pt.cmdscale(X)
    df_cmds = pd.DataFrame(data=X_cmds[0], index=df.index)
    times = sorted(list(set([int(x.split('_')[1]) for x in df_cmds.index.values])))
    mcds = []
    fig = plt.figure()
    for time in times:
        time_cmds = df_cmds[df_cmds.index.str.contains('_' + str(time))].as_matrix()
        mcds.append(pt.get_mean_centroid_distance(time_cmds, k = 5))
        time_null_mcd = df_null.loc[df_null['Time'] == time].MCD.values
        plt.scatter([int(time)]* len(time_null_mcd), time_null_mcd, c='#87CEEB', marker = 'o', s = 120, \
            edgecolors='none', linewidth = 0.6, alpha = 0.3)#, edgecolors='none')
    plt.scatter(times, mcds, c='#175ac6', marker = 'o', s = 120, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.9)#, edgecolors='none')

    plt.xlabel("Time (generations)", fontsize = 16)
    plt.ylabel("Mean centroid distance", fontsize = 16)
    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/permutation_scatter_good.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()

    #df_cmds_mcd = pt.get_mean_centroid_distance(df_cmds.as_matrix(), k = 5)

    #colors = np.linspace(min(times),max(times),len(times))
    #color_dict = dict(zip(times, colors))


#def get_mean_rbg(c1):


def example_gene_space():
    x = [2.5, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 2.5]
    y = [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4]
    gt = ['0000', '0001', '0010', '0100', '1000', '0011', '0101', '1001', \
                '0110', '1010', '1100', '0111', '1011', '1101', '1110', '1111']
    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    for i, x_i in enumerate(x):
        y_i = y[i]
        gt_i = gt[i]
        w1 = gt_i[0:2].count('1') / 3
        w2 = gt_i[2:].count('1') / 3
        new_color = pt.get_mean_colors('#FF3333', '#3333FF', w1, w2)
        plt.scatter(x_i, y_i, facecolors=new_color, edgecolors='k', marker = 'o', \
            s = 900, linewidth = 1, alpha = 0.7)
        plt.text(x_i, y_i, gt_i, color = 'k', ha = 'center', va = 'center')


    fig.savefig(pt.get_path() + '/figs/gene_space.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




#plot_mcd_pcoa_good()
#plot_pcoa('tenaillon')
#example_gene_space()
plot_permutation('tenaillon')
