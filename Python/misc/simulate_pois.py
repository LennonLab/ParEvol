from __future__ import division
import math, random, itertools
import numpy as np
import pandas as pd
import scipy.stats as stats
import parevol_tools as pt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import networkx as nx



def get_line_cov_matrix(n_rows, cov, var=1,alternate_cov_sign=False):
    row_list = []
    cov_sign_dict = {}
    for i in range(n_rows-1):
        #cov = np.random.normal(loc=0.0, scale=5.0)
        if i%2 == 1:
            if alternate_cov_sign == True:
                cov_sign_dict[str(i)+'_'+str(i+1)] = -cov
            else:
                cov_sign_dict[str(i)+'_'+str(i+1)] = cov
        else:
            cov_sign_dict[str(i)+'_'+str(i+1)] = cov

    for i in range(n_rows):
        list_i = [0] * n_rows
        list_i[i] = var
        if i == 0:
            list_i[i+1] = cov_sign_dict[str(i)+'_'+str(i+1)]
        elif 0 < i < (n_rows-1):
            list_i[i-1] = cov_sign_dict[str(i-1)+'_'+str(i)]
            list_i[i+1] = cov_sign_dict[str(i)+'_'+str(i+1)]
        else:
            list_i[i-1] = cov_sign_dict[str(i-1)+'_'+str(i)]
        row_list.append(list_i)
    print(row_list)
    return np.asarray(row_list)


def get_network_cov(cov = 1/9, var = 1):
    #df = pd.read_csv(pt.get_path() + '/data/disassoc_network_eq.txt', sep='\t',  header=None)
    #df = df.astype(int)
    ntwrk = np.loadtxt(pt.get_path() + '/data/disassoc_network_eq.txt', delimiter='\t')#, dtype='int')
    #print(np.mean(np.sum(ntwrk, axis =1)))
    ntwrk = ntwrk * cov
    np.fill_diagonal(ntwrk, var)

    # Gershgorin circle theorem sets limit on covariance
    # https://math.stackexchange.com/questions/2378428/how-to-create-a-positive-definite-covariance-matrix-from-an-adjacency-matrix
    graph = nx.barabasi_albert_graph(50, 5)
    graph_np = nx.to_numpy_matrix(graph)
    #print(np.sum(graph_np, axis =1))

    graph_np = graph_np * cov
    np.fill_diagonal(graph_np, 1)

    #print(np.linalg.eigvals(graph_np))
    #print(np.all(np.linalg.eigvals(graph_np) > 0))
    #graph_np = graph_np * 0.49
    #np.fill_diagonal(graph_np, 1)

    #print(ntwrk)
    print(np.linalg.eigvals(ntwrk))
    print(np.all(np.linalg.eigvals(ntwrk) > 0))




def get_pois_sample(lambda_, u):
    x = 0
    p = math.exp(-lambda_)
    s = p
    #u = np.random.uniform(low=0.0, high=1.0)
    while u > s:
         x = x + 1
         p  = p * lambda_ / x
         s = s + p
    return x


def get_count_pop(lambdas, cov):
    C =cov
    mult_norm = np.random.multivariate_normal(np.asarray([0]* len(lambdas)), C)#, tol=1e-6)
    mult_norm_cdf = stats.norm.cdf(mult_norm)
    counts = [ get_pois_sample(lambdas[i], mult_norm_cdf[i]) for i in range(len(lambdas))  ]

    return np.asarray(counts)


def get_block_cov(n_genes, var=1, pos_cov = 0.9, neg_cov= -0.9):
    cov = np.full((n_genes, n_genes), float(0))
    cutoff = int(n_genes/2)
    for i in range(n_genes):
        cov[i,i] = var
        for j in range(n_genes):
            if i < j:
                if ((i < cutoff) and ( j < cutoff)) or ((i >= cutoff) and ( j >= cutoff)):
                    cov[i,j] = pos_cov
                    cov[j,i] = pos_cov
                else:
                    cov[i,j] = neg_cov
                    cov[j,i] = neg_cov
    return cov


def run_block_cov_sims():
    df_out=open(pt.get_path() + '/data/simulations/cov_block_euc_pos_only.txt', 'w')
    n_pops=20
    n_genes=50
    lambda_genes = np.random.gamma(shape=3, scale=1, size=n_genes)
    df_out.write('\t'.join(['Cov', 'Iteration', 'z_score']) + '\n')
    #covs = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
    covs = [-0.9]
    for cov in covs:
        C = get_block_cov(n_genes, pos_cov = cov, neg_cov = cov)
        print(np.all(np.linalg.eigvals(C) > 0))
        print(C)
        for i in range(100):
            test_cov = np.stack( [get_count_pop(lambda_genes, cov= C) for x in range(n_pops)] , axis=0 )
            X = pt.hellinger_transform(test_cov)
            pca = PCA()
            pca_fit = pca.fit_transform(X)
            euc_dist = pt.get_mean_pairwise_euc_distance(pca_fit)
            sim_eucs = []
            for j in range(1000):
                X_j = pt.hellinger_transform(pt.random_matrix(test_cov))
                pca_fit_j = pca.fit_transform(X_j)
                sim_eucs.append( pt.get_mean_pairwise_euc_distance(pca_fit_j) )
            z_score = (euc_dist - np.mean(sim_eucs)) / np.std(sim_eucs)
            print(str(cov) , ' ', str(i), ' ', str(z_score))
            df_out.write('\t'.join([str(cov), str(i), str(z_score)]) + '\n')

    df_out.close()


def run_ntwrk_cov_sims(var = 1, cov = 0.25):
    df_out=open(pt.get_path() + '/data/simulations/cov_ntwrk_euc_pos_only_010.txt', 'w')
    n_pops=20
    n_genes=250
    lambda_genes = np.random.gamma(shape=3, scale=1, size=n_genes)
    df_out.write('\t'.join(['Cov', 'Iteration', 'z_score']) + '\n')
    C = np.loadtxt(pt.get_path() + '/data/modular_ntwrk_mu_010.txt', delimiter='\t')#, dtype='int')
    #print(np.mean(np.sum(ntwrk, axis =1)))
    C = C * cov
    np.fill_diagonal(C, var)
    for i in range(100):
        test_cov = np.stack( [get_count_pop(lambda_genes, cov= C) for x in range(n_pops)] , axis=0 )
        X = pt.hellinger_transform(test_cov)
        pca = PCA()
        pca_fit = pca.fit_transform(X)
        euc_dist = pt.get_mean_pairwise_euc_distance(pca_fit)
        sim_eucs = []
        for j in range(1000):
            X_j = pt.hellinger_transform(pt.random_matrix(test_cov))
            pca_fit_j = pca.fit_transform(X_j)
            sim_eucs.append( pt.get_mean_pairwise_euc_distance(pca_fit_j) )
        z_score = (euc_dist - np.mean(sim_eucs)) / np.std(sim_eucs)
        print(str(cov) , ' ', str(i), ' ', str(z_score))
        df_out.write('\t'.join([str(cov), str(i), str(z_score)]) + '\n')

    df_out.close()


def run_ba_ntwk_cov_sims():
    df_out=open(pt.get_path() + '/data/simulations/cov_ba_ntwrk_ev.txt', 'w')
    n_pops=100
    n_genes=50
    ntwk = nx.barabasi_albert_graph(n_genes, 2)
    ntwk_np = nx.to_numpy_matrix(ntwk)
    lambda_genes = np.random.gamma(shape=3, scale=1, size=n_genes)
    df_out.write('\t'.join(['Cov', 'Iteration', 'euc_z_score', 'euc_percent', 'eig_percent', 'mcd_percent_k1', 'mcd_percent_k3']) + '\n')
    covs = [0.05, 0.1, 0.15, 0.2]
    #covs = [0.2, 0.7]
    for cov in covs:
        C = ntwk_np * cov
        np.fill_diagonal(C, 1)
        #z_scores = []
        #eig_percents = []
        #euc_percents = []
        #centroid_percents_k1 = []
        #centroid_percents_k3 = []
        for i in range(1000):
            test_cov = np.stack( [get_count_pop(lambda_genes, cov= C) for x in range(n_pops)] , axis=0 )
            X = pt.hellinger_transform(test_cov)
            pca = PCA()
            pca_fit = pca.fit_transform(X)
            euc_dist = pt.get_mean_pairwise_euc_distance(pca_fit)
            euc_dists = []
            eig = pt.get_x_stat(pca.explained_variance_[:-1])
            mcd_k1 = pt.get_mean_centroid_distance(pca_fit, k = 1)
            mcd_k3 = pt.get_mean_centroid_distance(pca_fit, k = 3)
            eigs = []
            centroid_dists_k1 = []
            centroid_dists_k3 = []
            for j in range(1000):
                X_j = pt.hellinger_transform(pt.random_matrix(test_cov))
                #pca_j = PCA()
                #pca_fit_j = pca_j.fit_transform(X_j)
                pca_fit_j = pca.fit_transform(X_j)
                euc_dists.append( pt.get_mean_pairwise_euc_distance(pca_fit_j) )
                centroid_dists_k1.append(pt.get_mean_centroid_distance(pca_fit_j, k = 1))
                centroid_dists_k3.append(pt.get_mean_centroid_distance(pca_fit_j, k = 3))
                eigs.append( pt.get_x_stat(pca.explained_variance_[:-1]) )
                #eigs.append( pt.get_x_stat(pca_j.explained_variance_[:-1]) )
            z_score = (euc_dist - np.mean(euc_dists)) / np.std(euc_dists)
            euc_percent = len( [k for k in euc_dists if k < euc_dist] ) / len(euc_dists)
            eig_percent = len( [k for k in eigs if k < eig] ) / len(eigs)
            centroid_percent_k1 = len( [k for k in centroid_dists_k1 if k < mcd_k1] ) / len(centroid_dists_k1)
            centroid_percent_k3 = len( [k for k in centroid_dists_k3 if k < mcd_k3] ) / len(centroid_dists_k3)
            #eig_percents.append(eig_percent)
            #euc_percents.append(euc_percent)
            #z_scores.append(z_score)
            print(cov, i, z_score, euc_percent, eig_percent)
            df_out.write('\t'.join([str(cov), str(i), str(z_score), str(euc_percent), str(eig_percent), str(centroid_percent_k1), str(centroid_percent_k3)]) + '\n')

        #print(cov, np.all(np.linalg.eigvals(C) > 0), np.mean(z_scores))

    df_out.close()



def run_all_sims():
    df_out=open(pt.get_path() + '/data/simulations/cov_euc.txt', 'w')
    n_pops=20
    n_genes=50
    lambda_genes = np.random.gamma(shape=3, scale=1, size=n_genes)
    covs = [0.5, 0, -0.5]
    df_out.write('\t'.join(['Covariance', 'Iteration', 'z_score']) + '\n')
    for cov in covs:
        for i in range(100):
            print(str(cov) + ' '  + str(i))
            test_cov = np.stack( [get_count_pop(lambda_genes, cov= cov) for x in range(n_pops)] , axis=0 )
            X = pt.hellinger_transform(test_cov)
            pca = PCA()
            pca_fit = pca.fit_transform(X)
            euc_dist = pt.get_euclidean_distance(pca_fit)
            sim_eucs = []
            for j in range(1000):
                #if j % 100 == 0:
                #    print(j)
                X_j = pt.hellinger_transform(pt.random_matrix(test_cov))
                pca_fit_j = pca.fit_transform(X_j)
                sim_eucs.append( pt.get_euclidean_distance(pca_fit_j) )
            z_score = (euc_dist - np.mean(sim_eucs)) / np.std(sim_eucs)

            df_out.write('\t'.join([str(cov), str(i), str(z_score)]) + '\n')

    df_out.close()



def get_fig():
    df = pd.read_csv(pt.get_path() + '/data/simulations/cov_ba_ntwrk_ev.txt', sep='\t')
    x = df.Cov.values
    y = df.euc_z_score.values
    #print(np.mean(y))

    fig = plt.figure()
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    x_slope = np.linspace(0,1,1000)
    y_slope = intercept + ( slope *  x_slope)

    plt.scatter(x, y, c='#175ac6', marker = 'o', s = 70, \
        edgecolors='#244162', linewidth = 0.6, alpha = 0.2, zorder=2)#, edgecolors='none')
    plt.plot(x_slope, y_slope, c='k', lw =2)
    plt.axhline(y=0, color='red', lw=2, linestyle='--')
    plt.xlabel('Covariance')
    plt.ylabel('Z-score')

    plt.tight_layout()
    fig_name = pt.get_path() + '/figs/cov_ba_ntwrk_ev.png'
    fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
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


power_figs()
#run_ntwrk_cov_sims()
#get_fig()
#df1 = pd.read_csv(pt.get_path() + '/data/simulations/cov_ntwrk_euc_pos_only_025.txt', sep='\t')
#print(np.mean(df1.z_score.values))
#df2 = pd.read_csv(pt.get_path() + '/data/simulations/cov_ntwrk_euc_pos_only_015.txt', sep='\t')
#print(np.mean(df2.z_score.values))
#df3 = pd.read_csv(pt.get_path() + '/data/simulations/cov_ntwrk_euc_pos_only_010.txt', sep='\t')
#print(np.mean(df3.z_score.values))
#run_block_cov_sims()


#run_ba_ntwk_cov_sims()
