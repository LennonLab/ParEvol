import networkx as nx


graph = nx.barabasi_albert_graph(10, 2)
print(graph.nodes())
print(graph.edges())
