from __future__ import division
import numpy as np

matrix = np.asarray([ [0,0,1], [1,0,0], [1,1,0], [1,0,1]])

lengths = np.asarray([80,150,200])
mean_length =np.mean(lengths)
N_genes = len(lengths)
m_mean = np.sum(matrix, axis=0) / N_genes
print(matrix)
print(m_mean/ lengths)
print((matrix*(mean_length / lengths)) )
#print( matrix * np.log((matrix * (mean_length / lengths)) / m_mean) )
