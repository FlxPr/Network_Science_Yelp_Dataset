import pandas as pd
from settings import file_names
import networkx as nx
import community  # run pip install python-louvain
import json
import sknetwork as skn
from itertools import chain


def make_friends_graph():
    def get_friends_pairs(user_id, friends):
        return [(user_id, friend) for friend in friends.split(', ')] if type(friends) == str else None

    df = pd.read_csv(file_names['toronto_users'])
    g = nx.Graph()
    g.add_nodes_from(df.user_id.unique())
    g.add_edges_from(
        chain.from_iterable(df.apply(lambda row: get_friends_pairs(row['user_id'], row['friends']), axis=1).dropna())
    )
    return g


def make_user_business_bipartite_graph(weighted=True):
    df = pd.read_csv(file_names['toronto_reviews_without_text'])
    g = nx.Graph()
    g.add_nodes_from(df.user_id.unique(), bipartite=0)
    g.add_nodes_from(df.business_id.unique(), bipartite=1)
    if weighted:
        g.add_weighted_edges_from([(user, business, rating) for user, business, rating
                               in zip(df.user_id, df.business_id, df.rating)])
    else:
        g.add_edges_from([(user, business) for user, business, rating
                               in zip(df.user_id, df.business_id)])

    return g


def communities_friends(g: nx.Graph): # TODO complete function
    try:
        communities = json.load(open(file_names['toronto_communities']))
    except FileNotFoundError:
        print('Computing communities with Louvain Algorithm...')
        communities = community.best_partition(g)  # Calculate partitioning using Louvain Algorithm
        print('Saving community partition to {}'.format(file_names['toronto_communities']))
        with open(file_names['toronto_communities'], 'w+') as f:
            json.dump(communities, fp=f)

        pass


def bilouvain_sclustering():  # TODO complete function
    adjacency_matrix = nx.to_scipy_sparse_matrix(g)
    bilouvain = skn.clustering.BiLouvain()
    bilouvain.fit(adjacency_matrix)
