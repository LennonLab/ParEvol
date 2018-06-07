from __future__ import division
import numpy as np


class simulate_NK:
    def __init__(self, N, K, G, alpha):
        self.N = N
        self.K = K
        self.G = G
        # multiplicative, range from 0 to infinity
        self.alpha = alpha


    def get_undirected_graph(self):
        ####### warning
        ####### does not always return graph with equal number of edges for all nodes
        gene_dict = {}
        genes = np.random.choice(self.N, size=[self.G, int(self.N/self.G)], replace=False)
        for i, gene in enumerate(genes):
            for site in gene:
                gene_dict[site] = i
        sites = list(range(self.N))
        # we want an undirected graph, so keep track of the contributions
        interact_dict = {}
        for key, value in gene_dict.items():
            sites_key = list(sites)
            del sites_key[key]
            to_remove = [x for x in interact_dict if len(interact_dict[x]) >= self.K ]
            sites_key = [x for x in sites_key if x not in to_remove]
            if key not in interact_dict:
                sample_size = min(self.K, len(sites_key))
                if sample_size <= 0:
                    continue
                sites_K = np.random.choice(sites_key, size=sample_size, replace=False)
                interact_dict[key] = list(sites_K)
                for site_K in sites_K:
                    if site_K not in gene_dict:
                        interact_dict[site_K] = [key]
                    else:
                        interact_dict.setdefault(site_K, []).append(key)
            else:
                sample_size = min(self.K - len(interact_dict[key]), len(sites_key))
                if self.K - len(interact_dict[key]) > len(sites_key):
                    sample_size = len(sites_key)
                else:
                    sample_size = self.K - len(interact_dict[key])
                if sample_size <= 0:
                    continue
                sites_K = np.random.choice(sites_key, size=sample_size, replace=False)
                for site_K in sites_K:
                    interact_dict.setdefault(site_K, []).append(key)
                    interact_dict.setdefault(key, []).append(site_K)


        return interact_dict

    def get_directed_graph(self):
        site_dict = {}
        gene_dict = {}
        genes = np.random.choice(self.N, size=[self.G, int(self.N/self.G)], replace=False)
        for i, gene in enumerate(genes):
            gene_dict[i] = list(gene)
            for site in gene:
                site_dict[site] = i
        sites = list(range(self.N))
        interact_dict = {}
        # much simpler to code, since we want a directed graph
        for key, value in site_dict.items():
            sites_key = list(sites)
            sites_key.remove(key)
            gene_sites = list(gene_dict[value])
            gene_sites.remove(key)
            sites_outside_gene = [x for x in sites_key if x not in gene_sites]
            print(sites_outside_gene)
            print(gene_sites)
            # zip list? sort by value (left) and have gene name on right
            # then use that to multiply by alpha

            #p =
            sites_K = np.random.choice(sites_key, size=self.K, replace=False)
            interact_dict[key] = list(sites_K)
        print(interact_dict)


simulate_NK(4, 2, 2, 0.1).get_directed_graph()
