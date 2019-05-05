


def get_tenaillon_pca():
    df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_delta = pt.likelihood_matrix(df, 'Tenaillon_et_al').get_likelihood_matrix()
    X = pt.hellinger_transform(df_delta)
    pca = PCA()
    df_out = pca.fit_transform(X)

    fig = plt.figure()
    plt.axhline(y=0, color='k', linestyle=':', alpha = 0.8, zorder=1)
    plt.axvline(x=0, color='k', linestyle=':', alpha = 0.8, zorder=2)
    plt.scatter(0, 0, marker = "o", edgecolors='none', c = 'darkgray', s = 120, zorder=3)
    plt.scatter(df_out[:,0], df_out[:,1], marker = "o", edgecolors='#244162', c = '#175ac6', alpha = 0.6, s = 80, zorder=4)

    plt.xlim([-0.8,0.8])
    plt.ylim([-0.8,0.8])
    plt.xlabel('PCA 1 (' + str(round(pca.explained_variance_ratio_[0],3)*100) + '%)' , fontsize = 16)
    plt.ylabel('PCA 2 (' + str(round(pca.explained_variance_ratio_[1],3)*100) + '%)' , fontsize = 16)
    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/pca_tenaillon.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()





def ltee_convergence(alpha = 0.05, k = 5):
    df_path = pt.get_path() + '/data/Good_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    to_exclude = pt.complete_nonmutator_lines()
    to_exclude.append('p5')
    df_nonmut = df[df.index.str.contains('|'.join( to_exclude))]
    # remove columns with all zeros
    df_nonmut = df_nonmut.loc[:, (df_nonmut != 0).any(axis=0)]
    df_delta = pt.likelihood_matrix(df_nonmut, 'Good_et_al').get_likelihood_matrix()

    pca = PCA()
    df_out = pca.fit_transform(X)

    time_points = [ int(x.split('_')[1]) for x in df_nonmut.index.values]
    time_points_set = sorted(list(set([ int(x.split('_')[1]) for x in df_nonmut.index.values])))

    df_rndm_delta_out = pd.DataFrame(data=df_out, index=df_delta.index)
    mcds = []
    angles = []
    Ls = []
    for tp in time_points_set:
        df_rndm_delta_out_tp = df_rndm_delta_out[df_rndm_delta_out.index.str.contains('_' + str(tp))]
        mcds.append(pt.get_mean_centroid_distance(df_rndm_delta_out_tp.as_matrix(), k=k))
        angles.append(pt.get_mean_angle(df_rndm_delta_out_tp.as_matrix(), k=k))
        Ls.append(pt.get_euclidean_distance(df_rndm_delta_out_tp.as_matrix(), k=k))

    perm_path = pt.get_path() + '/data/Good_et_al/permute_PCA.txt'
    perm = pd.read_csv(perm_path, sep = '\t', header = 'infer', index_col = 0)
    perm_gens = np.sort(list(set(perm.Generation.tolist())))
    lower_ci_mcd = []
    upper_ci_mcd = []
    lower_ci_angle = []
    upper_ci_angle = []
    lower_ci_L = []
    upper_ci_L = []
    mean_mcd = []
    mean_angle = []
    mean_L = []
    for x in perm_gens:
        perm_x = perm.loc[perm['Generation'] == x]
        mcd_perm_x = np.sort(perm_x.MCD.tolist())
        angle_perm_x = np.sort(perm_x.mean_angle.tolist())
        L_perm_x = np.sort(perm_x.delta_L.tolist())

        mean_mcd_perm_x = np.mean(mcd_perm_x)
        mean_mcd.append(mean_mcd_perm_x)
        mean_angle_perm_x = np.mean(angle_perm_x)
        mean_angle.append(mean_angle_perm_x)
        mean_L_perm_x = np.mean(L_perm_x)
        mean_L.append(mean_L_perm_x)

        lower_ci_mcd.append(mean_mcd_perm_x - mcd_perm_x[int(len(mcd_perm_x) * alpha)])
        upper_ci_mcd.append(abs(mean_mcd_perm_x - mcd_perm_x[int(len(mcd_perm_x) * (1 - alpha))]))

        lower_ci_angle.append(mean_angle_perm_x - angle_perm_x[int(len(angle_perm_x) * alpha)])
        upper_ci_angle.append(abs(mean_angle_perm_x - angle_perm_x[int(len(angle_perm_x) * (1 - alpha))]))

        lower_ci_L.append(mean_L_perm_x - L_perm_x[int(len(L_perm_x) * alpha)])
        upper_ci_L.append(abs(mean_L_perm_x - L_perm_x[int(len(L_perm_x) * (1 - alpha))]))

    fig = plt.figure()

    plt.figure(1)
    plt.subplot(311)
    plt.errorbar(perm_gens, mean_mcd, yerr = [lower_ci_mcd, upper_ci_mcd], fmt = 'o', alpha = 0.5, \
        barsabove = True, marker = '.', mfc = 'k', mec = 'k', c = 'k', zorder=1)
    plt.scatter(time_points_set, mcds, c='#175ac6', marker = 'o', s = 70, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)#, edgecolors='none')

    #plt.ylabel("Mean \n centroid distance", fontsize = 10)
    plt.ylabel(r'$\left \langle \delta_{c}  \right \rangle$', fontsize = 12)

    plt.figure(1)
    plt.subplot(312)
    plt.errorbar(perm_gens, mean_angle, yerr = [lower_ci_angle, upper_ci_angle], fmt = 'o', alpha = 0.5, \
        barsabove = True, marker = '.', mfc = 'k', mec = 'k', c = 'k', zorder=1)
    plt.scatter(time_points_set, angles, c='#175ac6', marker = 'o', s = 70, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)#, edgecolors='none')

    #plt.ylabel("Standardized mean \n centroid distance", fontsize = 14)
    plt.ylabel(r'$\left \langle \theta \right \rangle$', fontsize = 12)

    plt.figure(1)
    plt.subplot(313)
    plt.errorbar(perm_gens, mean_L, yerr = [lower_ci_L, upper_ci_L], fmt = 'o', alpha = 0.5, \
        barsabove = True, marker = '.', mfc = 'k', mec = 'k', c = 'k', zorder=1)
    plt.scatter(time_points_set, Ls, c='#175ac6', marker = 'o', s = 70, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.5, zorder=2)#, edgecolors='none')

    plt.xlabel("Time (generations)", fontsize = 16)
    #plt.ylabel("Standardized mean \n centroid distance", fontsize = 14)
    plt.ylabel(r'$\left \langle  \left | \Delta L \right |\right \rangle$', fontsize = 12)

    fig.tight_layout()
    fig.savefig(pt.get_path() + '/figs/fig3.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()






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
