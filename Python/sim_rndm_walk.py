from __future__ import division
import math
import numpy as np
import scipy.stats as stats
from random import randint
import parevol_tools as pt
import random
import matplotlib.pyplot as plt

def walk_sim():
    L1 = 10
    L2 = 10
    s = 0.01
    subs = 10
    # key = gene1site_gene2site
    inter_dict = {}
    # assume genes are the same length
    for i in range(L1):
        inter_dict[str(i) + '_' + str(i)] = np.random.normal(-1, 1)

    seq_dict = {i:0 for i in range(L1 + L2)}
    #seq_g2_dict = {i:0 for i in range(L2)}
    F = 0
    sites = list(range(L1 + L2))
    counts_1 = 0
    counts_2 = 0
    for i in range(subs):
        #site_i = randint( 0, L1 + L2 -1 )
        site_i = random.choice(sites)
        sites.remove(site_i)
        seq_dict[site_i] = 1
        if site_i >= 10:
            site_i_partner = site_i - 10
            counts_2 += 1
            if seq_dict[site_i_partner] == 1:
                inter_i = inter_dict[str(site_i_partner)+'_'+str(site_i_partner)]
            else:
                inter_i = 0

        else:
            site_i_partner = site_i + 10
            counts_1 += 1
            if seq_dict[site_i_partner] == 1:
                inter_i = inter_dict[str(site_i)+'_'+str(site_i)]
            else:
                inter_i = 0

        F_new = F + s + inter_i
        #print(i, F, F_new, inter_i)
        if F_new > F:
            F = F_new
            if i == subs-1:
                return [counts_1,counts_2]
        else:
            return [counts_1,counts_2]
            break

diffs = []
for i in range(1000):
    result = walk_sim()
    diffs.append( result[0] - result[1]  )


fig = plt.figure()
plt.hist(diffs,bins=20, alpha=0.4, color = 'blue')
plt.xlabel('Count difference')
plt.ylabel('Number')

plt.tight_layout()
fig_name = pt.get_path() + '/figs/test_rndm_walk.png'
fig.savefig(fig_name, bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

print(np.std(diffs))
