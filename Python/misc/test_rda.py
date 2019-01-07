import numpy as np
import pandas as pd
#from skbio.stats.ordination import rda
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, Formula
import rpy2.robjects as robjects

# site-by-species
Y = np.array([[1, 0, 0, 0, 0, 0, 2, 4, 4],
               [0, 0, 0, 0, 0, 0, 5, 6, 1],
               [0, 1, 0, 0, 0, 0, 0, 2, 3],
               [11, 4, 0, 0, 8, 1, 6, 2, 0],
               [11, 5, 17, 7, 0, 0, 6, 6, 2],
               [9, 6, 0, 0, 6, 2, 10, 1, 4],
               [9, 7, 13, 10, 0, 0, 4, 5, 4],
               [7, 8, 0, 0, 4, 3, 6, 6, 4],
               [7, 9, 10, 13, 0, 0, 6, 2, 0],
               [5, 10, 0, 0, 2, 4, 0, 1, 3]])

X = np.asarray( [[0,0,0,0,0,1,1,1,1,1]] )

#X = np.array([[1.0, 0.0, 1.0, 0.0],
#               [2.0, 0.0, 1.0, 0.0],
#               [3.0, 0.0, 1.0, 0.0],
#               [4.0, 0.0, 0.0, 1.0],
#               [5.0, 1.0, 0.0, 0.0],
#               [6.0, 0.0, 0.0, 1.0],
#               [7.0, 1.0, 0.0, 0.0],
#               [8.0, 0.0, 0.0, 1.0],
#               [9.0, 1.0, 0.0, 0.0],
#               [10.0, 0.0, 0.0, 1.0]])

vegan = importr('vegan')
numpy2ri.activate()
nr_Y, nc_Y = Y.shape
Y_r = robjects.r.matrix(Y, nrow=nr_Y, ncol=nc_Y)
nr_X, nc_X = X.shape
X_r = robjects.r.matrix(X, nrow=nr_X, ncol=nc_X)
print(X_r)

 #dbrda(fish.db ~ ., as.data.frame(env.chem))
fmla = Formula(' ~ .')

rda_vegan = vegan.rda(formula = fmla, data = X_r)
print(rda_vegan)
numpy2ri.deactivate()
#vegan.rda()

#Y_pd = (pd.DataFrame(Y))
#X_pd = (pd.DataFrame(X.T))

#print(X.T.shape)

#rda_results = rda(Y_pd, X_pd, scale_Y=False, scaling=1)

#print(rda_results)
