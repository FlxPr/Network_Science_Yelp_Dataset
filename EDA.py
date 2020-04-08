import pandas as pd
from tqdm import tqdm
import json
import igraph
from settings import file_names
import data_reader
import matplotlib.pyplot as plt
import seaborn as sns
import graph
from collections import Counter
import networkx as nx


if __name__ == '__main__':
    social_net = graph.make_friends_graph()
    n_users = social_net.number_of_nodes()
    n_friendships = social_net.number_of_edges()
    print('Friend network: \nNumber of users: {n_users} \nNumber of friendships {n_friendships}'
          .format(n_friendships=n_friendships, n_users=n_users))

    print('Number of friends')
    friends_distribution = Counter((node_degree[1] for node_degree in social_net.degree))
    print(friends_distribution)
    alone_user_percentage = friends_distribution[0]/sum(friends_distribution.values())
    print('{:.1f}% of the {} users have no friend link'.format(alone_user_percentage * 100, n_users))

    # Analyze connected components
    connected_components = list(nx.connected_components(social_net))
    connected_components_sizes = Counter(map(len, connected_components))
    print('Connected components sizes')
    print(connected_components_sizes)

    # Filter graph to largest connected component
    social_net = social_net.subgraph(max(nx.connected_components(social_net), key=len))
