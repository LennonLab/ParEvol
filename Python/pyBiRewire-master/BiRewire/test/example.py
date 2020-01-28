import BiRewire as br
import numpy as np
import igraph as i

g=i.Graph().Barabasi(100)
r1=br.Rewiring(data=g,type_of_graph='undirected')
r1.rewire()
data=r1.monitoring()
rr1=br.Rewiring(data=np.array(r1.data.get_adjacency().data),type_of_array='adjacence')
rr1.analysis()
rr1=br.Rewiring(data=np.array(r1.data.get_edgelist()),type_of_array='edgelist_u')
r2=br.Rewiring(data=np.array(g.get_adjacency().data),type_of_array='adjacence')
r2.rewire()
r2.similarity()
r2.sampler('/home/test_birewire')
x = np.ascontiguousarray(np.random.randint(2, size=100).reshape(20,5),dtype='H')
r3=br.Rewiring(data=x,type_of_array='incidence')
r3.rewire()
g_b=br.incidence(x)
r4=br.Rewiring(data=g_b,type_of_graph='bipartite')
r4.rewire()




import matplotlib.pyplot as plt
r3.analysis(N=600)
plt.plot(r3.jaccard_index)
plt.yscale('log')
plt.xscale('log')
plt.show()

m,col,row=br.read_BRCA("BRCA.csv")
r=br.Rewiring(data=m,type_of_array='incidence')
r.rewire()
r.analysis()
