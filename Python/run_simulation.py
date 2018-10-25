from __future__ import division
import os, pickle
import numpy as np
import pandas as pd
import parevol_tools as pt
from collections import Counter
from sklearn.decomposition import PCA
from shapely.geometry.polygon import LinearRing


def get_pop_matrix(n_pops, n_genes, subs, probs, env):
    n_pops_dict = {}
    for i in range(n_pops):
        mutation_counts = np.random.choice(n_genes, size = subs, replace=True, p = probs)
        mutation_counts_dict = Counter(mutation_counts)

        n_pops_dict[env + '_' + str(i)] = mutation_counts_dict

    return n_pops_dict



n_genes = 20

sample_rates = np.random.gamma(shape = 3, scale = 1.0, size = n_genes)
gene_sample_sizes = list(range(2, n_genes, 5) )
#prob_union = gene_sample_sizes[4] / 0.1
#gene_sample = np.random.choice(n_genes, size = gene_sample_sizes[-1] , replace=False)
gene_sample = np.random.choice(n_genes, size = n_genes, replace=False)


to_reshuffle = sample_rates[gene_sample]
sample_rates = np.delete(sample_rates, gene_sample)
# reshuffle and append twice, once for each treatment
np.random.shuffle(to_reshuffle)
sample_rates_reshuffled_env1 = np.append(sample_rates, to_reshuffle, axis=None)
np.random.shuffle(to_reshuffle)
sample_rates_reshuffled_env2 = np.append(sample_rates, to_reshuffle, axis=None)

sample_rates_reshuffled_env1_sum = sum(sample_rates_reshuffled_env1)
sample_rates_reshuffled_env1_prob = [ (x/sample_rates_reshuffled_env1_sum)  for x in sample_rates_reshuffled_env1]
sample_rates_reshuffled_env2_sum = sum(sample_rates_reshuffled_env2)
sample_rates_reshuffled_env2_prob = [ (x/sample_rates_reshuffled_env2_sum)  for x in sample_rates_reshuffled_env2]


mutation_counts_env1 = get_pop_matrix(3, n_genes, 10, sample_rates_reshuffled_env1_prob, 'env1')
mutation_counts_env2 = get_pop_matrix(3, n_genes, 10, sample_rates_reshuffled_env2_prob, 'env2')

mutation_counts_df = pd.DataFrame.from_dict({**mutation_counts_env1, **mutation_counts_env2}, orient='index')
mutation_counts_df = mutation_counts_df.fillna(0)

X = pt.hellinger_transform(mutation_counts_df)
pca = PCA()
pca_fit = pca.fit_transform(X)
pca_fit_out = pd.DataFrame(data=pca_fit, index=mutation_counts_df.index)
groups = [ x.split('_')[0] for x in mutation_counts_df.index.tolist()]

check_separation = pt.get_ordiellipse(pca_fit, groups)
print(check_separation)


# then, perform dimension reduction
# generate confidence ellipse for first two dimensions
# check if they overlap




#mutation_counts_df.to_csv(pt.get_path() + '/test_count.txt', sep = '\t')
