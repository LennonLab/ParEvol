from __future__ import division
import numpy as np
import pandas as pd
import parevol_tools as pt
import matplotlib.pyplot as plt
from matplotlib import cm


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



def plot_permutation(dataset):
    if dataset == 'tenaillon':
        df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        df_delta = pt.likelihood_matrix(df, 'Tenaillon_et_al').get_likelihood_matrix()
        df_delta_bc = np.sqrt(pt.get_scipy_bray_curtis(df_delta.as_matrix()))
        df_delta_cmd = pt.cmdscale(df_delta_bc)[0]
        mcd = pt.get_mean_centroid_distance(df_delta_cmd)

        mcd_perm_path = pt.get_path() + '/data/Tenaillon_et_al/permute.txt'
        mcd_perm = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        mcd_perm_list = mcd_perm.MCD.values

        fig = plt.figure()
        plt.hist(mcd_perm_list, bins=30, histtype='stepfilled', normed=True, alpha=0.6, color='b')
        plt.axvline(mcd, color = 'red')
        plt.xlabel("Mean centroid distance")
        plt.ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(pt.get_path() + '/figs/permutation_hist_tenaillon.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
        plt.close()

    else:
        print('finish this function')


def plot_rndm_pcoa():
    # test, will get rid of this later
    df_path = pt.get_path() + '/data/Good_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    to_exclude = pt.complete_nonmutator_lines()
    to_exclude.append('p5')
    df_nonmut = df[df.index.str.contains('|'.join( to_exclude))]

    df_rndm = pd.DataFrame(data=pt.random_matrix(df_nonmut.as_matrix()), index=df_nonmut.index, columns=df_nonmut.columns)
    df_rndm_delta = pt.likelihood_matrix(df_rndm, 'Good_et_al').get_likelihood_matrix()
    #df_rndm_delta_bc = np.sqrt(pt.get_scipy_bray_curtis(df_rndm_delta.as_matrix()))
    #df_rndm_delta_cmd = pt.cmdscale(df_rndm_delta_bc)[0]

    # square root transformation to remove negative eigenvalues
    X = np.sqrt(pt.get_scipy_bray_curtis(df_rndm_delta.as_matrix()))
    #X = pt.get_scipy_bray_curtis(df.as_matrix())
    X_cmds = pt.cmdscale(X)
    df_cmds = pd.DataFrame(data=X_cmds[0], index=df_rndm.index)
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
    fig.savefig(pt.get_path() + '/figs/pcoa_good_rndm.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()

#plot_rndm_pcoa()
