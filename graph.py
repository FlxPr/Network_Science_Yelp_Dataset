import pandas as pd
from settings import file_names
import networkx as nx
from itertools import chain
import igraph

def make_friends_graph(library='networkx'):
    """
    :return: social network graph
    """
    def get_friends_pairs(user_id, friends):
        return [(user_id, friend) for friend in friends.split(', ')] if type(friends) == str else None

    df = pd.read_csv(file_names['toronto_users'])

    if library is 'networkx':
        social_network = nx.Graph()
        social_network.add_nodes_from(df.user_id.unique())
        social_network.add_edges_from(
            chain.from_iterable(df.apply(lambda row: get_friends_pairs(row['user_id'], row['friends']), axis=1).dropna())
        )
        return social_network
    elif library is 'igraph':
        social_network = igraph.Graph()
        social_network.add_vertices(df.user_id.unique())
        social_network.add_edges(chain.from_iterable(
                df.apply(lambda row: get_friends_pairs(row['user_id'], row['friends']), axis=1).dropna())
        )
        return social_network
    else:
        raise ValueError('Please use either "networkx" or "igraph" as library')

def make_user_business_bipartite_graph(weighted=False, minimum_rating=4):
    """
    :param weighted: assign rating as user-business edge weight
    :param minimum_rating: minimum rating to create user-business edge
    :return: user-business interaction graph
    """
    if minimum_rating <= 5:
        raise ValueError('Minimum rating must be less than 6')

    df = pd.read_csv(file_names['toronto_reviews_without_text'])
    df = df[df.rating >= minimum_rating]

    review_network = nx.Graph()
    review_network.add_nodes_from(df.user_id.unique(), bipartite=0)
    review_network.add_nodes_from(df.business_id.unique(), bipartite=1)

    if weighted:
        review_network.add_weighted_edges_from([(user, business, rating) for user, business, rating
                               in zip(df.user_id, df.business_id, df.rating)])
    else:
        review_network.add_edges_from([(user, business) for user, business, rating
                               in zip(df.user_id, df.business_id)])

    return review_network

