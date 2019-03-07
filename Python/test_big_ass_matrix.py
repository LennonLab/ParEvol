import parevol_tools as pt
import numpy as np
import run_simulations as rs
import scipy.stats as stats
import os

G = 50
N = 50
lambda_genes_null = np.asarray([1] * G)
test_cov_neutral = np.stack( [pt.get_count_pop(lambda_genes_null, C= np.identity(G)) for x in range(N)] , axis=0 )


rndm = pt.get_random_matrix(test_cov_neutral)



x = [1,2,3,4]
y = [10,12,20,19]

#rho,p_value = stats.spearmanr(x,y)
#print(rho,p_value)

#print(rs.run_ba_cov_lampbda_edge_sims("file_path", 4))
#out_name = os.path.expanduser("~/GitHub/ParEvol") + '/data/simulations/test_F.txt'
#rs.two_treats_sim(out_name)

#print(pt.get_ba_cov_matrix(20, 0.2, cov2=0.1))


#print(pt.get_correlated_rndm_ntwrk(50, m=2, rho=0.3, rho2=None))
#print("old")
#for x in range(10):
#    print(pt.get_correlated_rndm_ntwrk_old(50, m=2, rho=0.3)[1])
#print("new")
#for x in range(10):
#    print(pt.get_correlated_rndm_ntwrk(50, m=2, rho=0.3)[1])

_1, _2,_3,_4 = pt.get_correlated_rndm_ntwrk_test_version(50, m=2, rho=0.2, rho2=0)
print(_2, _4)
