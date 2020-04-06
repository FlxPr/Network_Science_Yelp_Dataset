import pandas as pd
import numpy as np
from settings import file_names
import igraph
import networkx as nx
from networkx.algorithms import community as comnx
import community  # pip install python-louvain
import json
import sknetwork as skn

df = pd.read_csv(file_names['toronto_reviews_without_text'])

# NETWORKX
g = nx.Graph()
g.add_nodes_from(df.user_id.unique(), bipartite=0)
g.add_nodes_from(df.business_id.unique(), bipartite=1)

g.add_weighted_edges_from([(user, business, rating) for user, business, rating
                           in zip(df.user_id, df.business_id, df.rating)])


adjacency_matrix = nx.to_scipy_sparse_matrix(g)
bilouvain = skn.clustering.BiLouvain()
bilouvain.fit(adjacency_matrix)


try:
    communities = json.load(open(file_names['toronto_communities']))
except FileNotFoundError:
    print('Computing communities with Louvain Algorithm...')
    communities = community.best_partition(g)  # Calculate partitioning using Louvain Algorithm
    print('Saving community partition to {}'.format(file_names['toronto_communities']))
    with open(file_names['toronto_communities'], 'w+') as f:
        json.dump(communities, fp=f)



# IGRAPH
# g = igraph.Graph.TupleList([(user, business, weight) for user, business, weight
#                             in zip(df.user_id, df.business_id, df.rating)], weights=True)
#
# communities = g.community_leiden(objective_function='modularity')
