from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

def get_kde(array):
    grid_ = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.1, 10, 50)},
                    cv=20) # 20-fold cross-validation
    grid_.fit(array[:, None])
    x_grid_ = np.linspace(0, 2.5, 1000)
    kde_ = grid_.best_estimator_
    pdf_ = np.exp(kde_.score_samples(x_grid_[:, None]))
    pdf_ = [x / sum(pdf_) for x in pdf_]

    return [x_grid_, pdf_, kde_.bandwidth]

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


def plot_mcd_pcoa_good():
    df_path = pt.get_path() + '/data/Good_et_al/gene_by_pop_delta.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_null_path = pt.get_path() + '/data/Good_et_al/permute_PCA.txt'
    df_null = pd.read_csv(df_null_path, sep = '\t', header = 'infer', index_col = 0)
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
        time_null_mcd = df_null.loc[df_null['Generation'] == time].MCD.values
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







def multiplicity_hist():
    df_path = pt.get_path() + '/data/Good_et_al/gene_by_pop_delta.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    to_exclude = pt.complete_nonmutator_lines()
    df = df[df.index.str.contains('|'.join( to_exclude))]


def draw_graph(graph, labels=None, graph_layout='shell',
               node_size=12, node_color='blue', node_alpha=0.2,
               node_text_size=10,
               edge_color='blue', edge_alpha=0.3, edge_tickness=0.5,
               edge_text_pos=0.3,
               text_font='sans-serif'):

    # create networkx graph
    G=nx.Graph()

    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    # these are different layouts for the network you may try
    # shell seems to work best
    if graph_layout == 'spring':
        graph_pos=nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos=nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos=nx.random_layout(G)
    else:
        graph_pos=nx.shell_layout(G)

    # draw graph
    nx.draw_networkx_nodes(G,graph_pos,node_size=node_size,
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G,graph_pos,width=edge_tickness,
                           alpha=edge_alpha,edge_color=edge_color)
    #nx.draw_networkx_labels(G, graph_pos,font_size=node_text_size,
    #                        font_family=text_font)

    if labels is None:
        labels = range(len(graph))

    edge_labels = dict(zip(graph, labels))
    #nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels,
    #                             label_pos=edge_text_pos)
    # show graph
    #plt.show()
    plt.savefig(pt.get_path() + '/figs/network_tenaillon.png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)




def plot_network():
    df_path = pt.get_path() + '/data/Tenaillon_et_al/gene_by_pop.txt'
    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    df_delta = pt.likelihood_matrix(df, 'Tenaillon_et_al').get_likelihood_matrix()
    df_delta = df_delta.loc[:, (df_delta != float(0)).any(axis=0)]
    adjacency_matrix = pt.get_adjacency_matrix(df_delta.as_matrix())
    labels = df_delta.columns.values

    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    #gr = nx.Graph()
    #print(list(edges))
    #gr.add_edges_from(edges)
    #print(list(edges))
    print(len(list(edges)))
    draw_graph(list(edges), labels = labels)

    #nx.draw(gr, node_size=500, labels=labels, with_labels=True)
