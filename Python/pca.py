from __future__ import division
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD
import  matplotlib.pyplot as plt
from matplotlib import cm
import parevol_tools as pt


def get_pca():
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
    plt.scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=1)
    for pop in pt.complete_nonmutator_lines():
        pop_df_pca = df_pca[df_pca.index.str.contains(pop)]
        c_list = [ color_dict[int(x.split('_')[1])] for x in pop_df_pca.index.values]
        if  pt.nonmutator_shapes()[pop] == 'p2':
            size == 50
        else:
            size = 80
        plt.scatter(pop_df_pca.as_matrix()[:,0], pop_df_pca.as_matrix()[:,1], \
        c=c_list, cmap = cm.Blues, vmin=min(times), vmax=max(times), \
        marker = pt.nonmutator_shapes()[pop], s = size, edgecolors='#244162', linewidth = 0.6)#, edgecolors='none')
    c = plt.colorbar()
    c.set_label("Generations")
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



def test_biplot():
    df = pd.read_csv(pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop_delta.txt', sep = '\t', header = 'infer', index_col = 0)
    X = pt.hellinger_transform(df)
    pca = PCA()
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(data=X_pca, index=df.index)
    #df_pca_coeff = pd.DataFrame(data=np.transpose(pca.components_[0:2, :]), index=df.columns)
    pt.plot_eigenvalues(pca.explained_variance_ratio_, file_name = 'pca_biplot_tenaillon_eigen')

    score = X_pca[:,0:2]
    # ordered wrt df.columns
    coeff = np.transpose(pca.components_[0:2, :])

    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())

    fig = plt.figure()
    plt.axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)
    plt.axvline(x=0, color='k', linestyle=':', alpha = 0.8, zorder=2)
    plt.scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=3)
    plt.scatter(xs * scalex,ys * scaley, marker = "o", edgecolors='none', c = 'forestgreen', alpha = 0.6, s = 80, zorder=4)
    for i in range(n):
        if (coeff[i,0] < 0.1) and (coeff[i,1] < 0.1):
            continue
        print(coeff[i,0], coeff[i,1])
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'k',alpha = 0.8, zorder=4)
        #if labels is None:
        #    plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        #else:
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, df.columns.values[i], color = 'k', ha = 'center', va = 'center', zorder=5)

    plt.xlabel('PCA 1 (' + str(round(pca.explained_variance_ratio_[0],3)*100) + '%)' , fontsize = 16)
    plt.ylabel('PCA 2 (' + str(round(pca.explained_variance_ratio_[1],3)*100) + '%)' , fontsize = 16)
    plt.ylim([-0.9,0.9])
    plt.xlim([-0.9,0.9])
    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/pca_test_biplot.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


def test_good_biplot():
    df = pd.read_csv(pt.get_path() + '/data/Good_et_al/gene_by_pop_delta.txt', sep = '\t', header = 'infer', index_col = 0)

    to_exclude = pt.complete_nonmutator_lines()
    df = df[df.index.str.contains('|'.join( to_exclude))]

    X = pt.hellinger_transform(df)
    pca = PCA()
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(data=X_pca, index=df.index)
    pt.plot_eigenvalues(pca.explained_variance_ratio_, file_name = 'pca_biplot_good_eigen')

    score = X_pca[:,0:2]
    # ordered wrt df.columns
    coeff = np.transpose(pca.components_[0:2, :])

    times = sorted(list(set([int(x.split('_')[1]) for x in df.index.values])))
    colors = np.linspace(min(times),max(times),len(times))
    color_dict = dict(zip(times, colors))

    fig = plt.figure()
    plt.axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)
    plt.axvline(x=0, color='k', linestyle=':', alpha = 0.8, zorder=2)
    plt.scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=1)
    for pop in pt.complete_nonmutator_lines():
        pop_df_pca = df_pca[df_pca.index.str.contains(pop)]
        c_list = [ color_dict[int(x.split('_')[1])] for x in pop_df_pca.index.values]
        if  pt.nonmutator_shapes()[pop] == 'p2':
            size == 50
        else:
            size = 80
        plt.scatter(pop_df_pca.as_matrix()[:,0], pop_df_pca.as_matrix()[:,1], \
        c=c_list, cmap = cm.Blues, vmin=min(times), vmax=max(times), \
        marker = pt.nonmutator_shapes()[pop], s = size, edgecolors='#244162', linewidth = 0.6)#, edgecolors='none')

    n = coeff.shape[0]
    for i in range(n):
        if (coeff[i,0] < 0.1) and (coeff[i,1] < 0.1):
            continue
        print(coeff[i,0], coeff[i,1])
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'k',alpha = 0.8, zorder=4)
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, df.columns.values[i], color = 'k', ha = 'center', va = 'center', zorder=5)

    c = plt.colorbar()
    c.set_label("Generations")
    plt.xlim([-0.8,0.8])
    plt.ylim([-0.8,0.8])
    plt.xlabel('PCA 1 (' + str(round(pca.explained_variance_ratio_[0],3)*100) + '%)' , fontsize = 16)
    plt.ylabel('PCA 2 (' + str(round(pca.explained_variance_ratio_[1],3)*100) + '%)' , fontsize = 16)
    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/pca_biplot_good.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




#get_pca()
test_biplot()
#test_good_biplot()
