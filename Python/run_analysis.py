from __future__ import division
import numpy as np
import pandas as pd
import parevol_tools as pt
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as clr


test_array = np.array([[1, 2], [3, 4], [3, 0]])


def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return clr.LinearSegmentedColormap('colormap',cdict,1024)

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
        fig.savefig(pt.get_path() + '/figs/test_pcoa.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
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

        #fig = plt.figure()
        #plt.hist(iterations, bins=30, histtype='stepfilled', normed=True, alpha=0.6, color='b')
        #plt.axvline(mcd, color = 'red')
        #plt.xlabel("Mean centroid distance")
        #plt.ylabel("Frequency")
        #fig.tight_layout()
        #fig.savefig(pt.get_path() + '/figs/permutation_hist.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
        #plt.close()

    elif dataset == 'good':
        df_path = pt.get_path() + '/data/Good_et_al/gene_by_pop.txt'
        df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
        # exclude p5 since it doesn't move in ordination space
        to_exclude = pt.complete_nonmutator_lines()
        to_exclude.append('p5')
        df_nonmut = df[df.index.str.contains('|'.join( to_exclude))]
        time_points = set([ int(x.split('_')[1]) for x in df_nonmut.index.values])
        #for time_point in time_points:
        #    df_time_point = df_nonmut[df_nonmut.index.str.contains(str(time_point))]
        df_nonmut_array = df_nonmut.as_matrix()
        df_rndm = pd.DataFrame(data=pt.random_matrix(df_nonmut_array), index=df_nonmut.index, columns=df_nonmut.columns)
        df_rndm_delta = pt.likelihood_matrix(df_rndm, 'Good_et_al').get_likelihood_matrix()

        df_rndm_delta_bc = np.sqrt(pt.get_bray_curtis(df_rndm_delta.as_matrix()))
        df_rndm_delta_cmd = pt.cmdscale(df_rndm_delta_bc)[0]

        #print(pt.get_mean_centroid_distance(df_rndm_delta_cmd, groups = ))



#print(test_array)
#print(pt.random_matrix(test_array))
#permutation_analysis(1000)
#plot_pcoa()
#permutation_analysis(10, dataset = 'good
plot_pcoa('good')

#test_array = np.array([[1.5, 2], [3.6, 4], [3, 0],[3, 4]])
#print(pt.get_bray_curtis(test_array))
#print(pt.get_scipy_bray_curtis(test_array))
