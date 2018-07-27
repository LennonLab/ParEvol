from __future__ import division
import numpy as np
import parevol_tools as pt
import pandas as pd


class simulate_NK:
    def __init__(self, N, K, G, alpha, mu, w_increment = 1):
        self.N = N
        self.K = K
        self.G = G
        # multiplicative, can be -1, 0, or 1
        self.alpha = alpha
        self.mu = mu
        self.w_increment = w_increment

    def neighbors(self, genotype, genotypes, only_previous = False):
        if only_previous == True:
            # keep only the previous neighbor
            return [g for g in genotypes if (abs(g ^ genotype).sum() == 1) and (np.sum(g) < np.sum(genotype))]
        else:
            return [g for g in genotypes if abs(g ^ genotype).sum() == 1]


    def int2bits(self, k):
        x = list(map(int, bin(k)[2:]))
        pad = self.N - len(x)
        x = [0]*pad + x
        return x


    def bits2int(self, bits_list):
        #out_list = []
        #for item in bits_list:
        #    out_list.append(''.join([str(int(i == True))  for i in item]))
        #return out_list
        return ''.join([str(int(i == True))  for i in bits_list])


    def all_genotypes(self):
        return np.array([self.int2bits(k) for k in range(2**self.N)], dtype=bool)


    def fitness_i(self, genotype, i, contribs, mem):
        key = tuple(zip(contribs[i], genotype[contribs[i]]))

        #print(key)
        if key not in mem:
            #mem[key] = np.random.uniform(0, 1)
            mem[key] = np.random.laplace(self.mu, scale = 1)
        return mem[key]



    def fitness(self, genotype, contribs, mem):
        #print(genotype, contribs)
        return np.mean([
            self.fitness_i(genotype, i, contribs, mem) # ω_i
            for i in range(len(genotype))
        ])


    def get_directed_graph(self):
        site_dict = {}
        gene_dict = {}
        genes = np.random.choice(self.N, size=[self.G, int(self.N/self.G)], replace=False)
        for i, gene in enumerate(genes):
            gene_dict[i] = list(gene)
            for site in gene:
                site_dict[site] = i
        sites = list(range(self.N))
        contribs = {}
        # much simpler to code, since we want a directed graph
        for key, value in site_dict.items():
            sites_key = list(sites)
            sites_key.remove(key)
            gene_sites = list(gene_dict[value])
            gene_sites.remove(key)
            sites_outside_gene = [x for x in sites_key if x not in gene_sites]
            site_dict_copy = site_dict.copy()
            del site_dict_copy[key]
            site_dict_copy_lst = list(site_dict_copy.items())
            #p = [ (1 / len(site_dict_copy_lst)) * self.alpha if x[1] == value else (1 / len(site_dict_copy_lst)) for x in site_dict_copy_lst ]
            if self.alpha == -1:
                p = [ 0 if x[1] == value else 1 for x in site_dict_copy_lst ]
            elif self.alpha == 1:
                p = [ 1 if x[1] == value else 0 for x in site_dict_copy_lst ]
            elif self.alpha == 0:
                p = [ (1 / len(site_dict_copy_lst)) for x in site_dict_copy_lst ]
            p_ = [x / sum(p) for x in p]
            sites_K = np.random.choice(sites_key, size=self.K, replace=False, p =p_)
            contribs[key] = list(sites_K) #+ [key]

        print(contribs)

        fitness_mem = {}
        max_genotypes_and_w = []
        max_genotypes_w = []
        genotypes = self.all_genotypes()
        #fix it so intercation with itself is pulled from a uniform dist
        #ws = [self.fitness(g, contribs, fitness_mem) + (len(np.where(g == True)[0]) * self.w_increment) for g in genotypes]
        #ws = [self.fitness(g, contribs, fitness_mem) + np.random.uniform(0, 1) for g in genotypes]
        max_w = max(ws)
        min_w = min(ws)
        for i, genotype in enumerate(genotypes):
            print(genotype)
            #wi = (self.fitness(genotype, contribs, fitness_mem) + (len(np.where(genotype == True)[0]) * self.w_increment) - min_w) / (max_w - min_w)
            #wi = (self.fitness(genotype, contribs, fitness_mem) + np.random.uniform(0, 1) - min_w) / (max_w - min_w)

            maximum = True
            minimum = True
            for g in self.neighbors(genotype, genotypes):
                #w = (self.fitness(g, contribs, fitness_mem)+ (len(np.where(g == True)[0]) * self.w_increment) - min_w) / (max_w - min_w)
                #w = (self.fitness(g, contribs, fitness_mem)+ np.random.uniform(0, 1) - min_w) / (max_w - min_w)
                #print(w)
                #print(w)
                if w > wi:
                    maximum = False
                if w < wi:
                    minimum = False
            if maximum:
                max_genotypes_and_w.append((genotype, wi))

        mut_count_in_genes = {}
        for i in range(self.G):
            mut_count_in_genes[i] = 0
        print(max_genotypes_and_w)
        max_max_genotypes_and_w = [x for x in max_genotypes_and_w if x[1] == float(1)][0]
        #for genotype in max_max_genotypes_and_w:
        #    mut_sites = np.where(genotype[0] == True)[0]
        #    for mut_site in mut_sites:
        #        mut_count_in_genes[site_dict[mut_site]] += 1
        #mut_count_in_genes = {k: v / len(max_max_genotypes_and_w) for k, v in mut_count_in_genes.items()}
        print(max_genotypes_and_w)
        for mut_site in np.where(max_max_genotypes_and_w[0] == True)[0]:
            mut_count_in_genes[site_dict[mut_site]] += 1
        return mut_count_in_genes


#test = simulate_NK(4, 2, 2, 100, -2).get_directed_graph()
#for key, value in test.items():
#    print(key, value)
#print(test)

def simulate(N, Ks, G, alphas, mu, iter = 100):
    #alphas and mus is a list
    df_out = open(pt.get_path() + '/data/simulations/test.txt', 'w')
    header = ['N', 'K', 'Gene', 'Alpha', 'Mu', 'Muts', 'Iter']
    df_out.write('\t'.join(header) + '\n')
    for K in Ks:
        for alpha in alphas:
            for i in range(iter):
                print('K = ' + str(K), 'alpha = ' + str(alpha), 'iter = '+str(i))
                sim_gene_dict = simulate_NK(N, K, G, alpha, mu).get_directed_graph()
                for key, value in sim_gene_dict.items():
                    sim_out = [N, K, key, alpha, mu, value, i]
                    sim_out = [str(x) for x in sim_out]
                    df_out.write('\t'.join(sim_out) + '\n')

    df_out.close()


print(simulate_NK(8, 2, 2, 1, -6).get_directed_graph())

#simulate(12, [3,6,9], 3, [0.01,1,100], -2, iter = 100)
#df = pd.read_csv(pt.get_path() + '/data/simulations/test.txt' , sep = '\t')
#df_001 = df.loc[(df['K'] == 3) & (df['Alpha'] == 100)]
#print(df_001.loc[df_001['Gene'] == 2])
