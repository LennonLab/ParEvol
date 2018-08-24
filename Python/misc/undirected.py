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
