from __future__ import division
import math
import numpy as np
import scipy.stats as stats


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


C_p5 = np.asarray( [ [1, 0.5, 0.5], [0.5, 1, 0.5] , [0.5, 0.5, 1] ] )
C_m5 = np.asarray( [ [1, -0.5, -0.5], [-0.5, 1, -0.5] , [-0.5, -0.5, 1] ] )
C_0 = np.asarray( [ [1, 0, 0], [0, 1, 0] , [0, 0, 1] ] )
Cs = [C_p5, C_m5, C_0]
lambdas=[2,2,2]
for C in Cs:
    diffs_pdf = []
    diffs_cdf = []
    diffs_count = []
    for i in range(1000):
        mult_norm = np.random.multivariate_normal(np.asarray([0]* len(lambdas)), C)
        diffs_pdf.append( abs( mult_norm[0] - mult_norm[1]  ) )
        mult_norm_cdf = stats.norm.cdf(mult_norm)
        diffs_cdf.append( abs( mult_norm_cdf[0] - mult_norm_cdf[1]  ) )
        lambda_u = list(zip(lambdas, mult_norm_cdf))
        counts = [get_pois_sample(x[0], x[1]) for x in  lambda_u]

        if sum(counts) > 0:
            diffs_count.append( abs( counts[0] - counts[1] ) / sum( counts )   )

    print(C[0,1], np.mean(diffs_pdf))
    print(C[0,1], np.mean(diffs_cdf))
    print(C[0,1], np.mean(diffs_count))


#lambdas = np.random.gamma(shape=3, scale=1, size=2)

#print(mult_norm)
#mult_norm_cdf = stats.norm.cdf(mult_norm)
