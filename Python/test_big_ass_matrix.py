import parevol_tools as pt
import numpy as np
import run_simulations as rs
import scipy.stats as stats

G = 50
N = 50
neutral_ = 11
lambda_genes_null = np.asarray([neutral_] * G)
test_cov_neutral = np.stack( [pt.get_count_pop(lambda_genes_null, C= np.identity(G)) for x in range(N)] , axis=0 )


rndm = pt.get_random_matrix(test_cov_neutral)

#print(test_cov_neutral.sum(axis=0))

#print(rndm.sum(axis=0))
cov=0.2
#test_assoc, rho = pt.get_correlated_rndm_ntwrk(n_genes=50, cov =cov)
#C = test_assoc * cov
#np.fill_diagonal(C, 1)
#print( np.all(np.linalg.eigvals(C) > 0) )

#out_name = '/Users/WRShoemaker/GitHub/ParEvol/data/simulations/text_rho.txt'
#rs.run_ba_cov_rho_sims(out_name, covs = [0.2], rhos=[0.2], shape=1, scale=1, G = 50, N = 50, iter1=10, iter2=10)

#run_ba_cov_sims_N_out = '/Users/WRShoemaker/GitHub/ParEvol/data/simulations/' + 'ba_cov_N_sims_test' + '.txt'
#rs.run_ba_cov_sims(gene_list=[50], pop_list=[8],
#        out_name = run_ba_cov_sims_N_out, covs = [0.2], iter1=10, iter2=100)

x = [1,2,3,4]
y = [10,12,20,19]

rho,p_value = stats.spearmanr(x,y)
#print(rho,p_value)

#print(rs.run_ba_cov_lampbda_edge_sims("file_path", 4))

rs.two_treats_sim()
