import networkx as nx
from concurrent.futures import ProcessPoolExecutor
from node_embeddings import get_network

def get_diam(g):
    def diameter_of_subgraph(nodes):
        H = g.subgraph(nodes)
        return nx.diameter(H)

    components = list(nx.connected_components(g))

    with ProcessPoolExecutor() as ex:
        diameters = list(ex.map(diameter_of_subgraph, components))

    return diameters

import pandas as pd
nodes = pd.read_csv('../data/graphs/only_connected/nodes.csv')
edges = pd.read_csv('../data/graphs/only_connected/edges.csv')

g = get_network(nodes, edges)

print('made network')

print(get_diam(g))