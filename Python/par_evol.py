from __future__ import division
import os
import numpy as np

mydir = os.path.expanduser("~/GitHub/ParEvol/")

# start with one gene, the NK model

def int2bits(k, N):
    x = list(map(int, bin(k)[2:]))
    pad = N - len(x)
    x = [0]*pad + x
    return x

def all_genotypes(N):
    return np.array([int2bits(k, N) for k in range(2**N)], dtype=bool)

N = 4
K = 1

genotypes = all_genotypes(N)
print(genotypes)
#print(list(map(int, bin(5)[2:])))

# contribs = a dictionary of the K neighbors that each site interacts with
 
contribs = {
        i: sorted(np.random.choice(
            [n for n in range(N) if n != i],
            K,
            replace=False
        ).tolist() + [i])
        for i in range(N)
    }

print(contribs)
