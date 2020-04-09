import pandas as pd
from tqdm import tqdm
import json
import igraph
from settings import file_names
import data_reader
import matplotlib.pyplot as plt
import seaborn as sns
import graph
import numpy as np
from collections import Counter
import networkx as nx
import powerlaw
import community


def plot_powerlaw_fit(node_degrees):
    powerlaw_fit = powerlaw.Fit(node_degrees)

    # Plot power law
    friends_distribution = Counter(node_degrees)
    fig, ax = plt.subplots()
    ax.scatter(friends_distribution.keys(), friends_distribution.values())

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.35, 0.95, 'Power law degree estimate: {:.2f}'.format(powerlaw_fit.power_law.alpha),
            transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    plt.title('Social network degree distribution')
    plt.xlabel('Number of friends')
    plt.ylabel('Count')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.show()


def community_cumulative_plot(communities: dict):
    community_counts = Counter(communities.values())
    community_counts = sorted(community_counts.values(), reverse=True)
    cumulative_community_counts = list(map(lambda x: x/sum(community_counts), np.cumsum(community_counts)))
    plt.plot(cumulative_community_counts)
    plt.title('Cumulative community appartenance')
    plt.grid()
    plt.xlabel('Number of communities')
    plt.ylabel('Percentage of users represented')
    plt.show()


if __name__ == '__main__':
    social_net = graph.make_friends_graph()
    n_users = social_net.number_of_nodes()
    n_friendships = social_net.number_of_edges()
    print('Friend network: \nNumber of users: {n_users} \nNumber of friendships {n_friendships}'
          .format(n_friendships=n_friendships, n_users=n_users))

    # Fit a power-law to the data
    node_degrees = [node_degree[1] for node_degree in social_net.degree]
    plot_powerlaw_fit(node_degrees)

    # Use louvain algorithm to maximize modularity in community detection
    print('Running Louvain algorithm for community detection...')
    communities = community.best_partition(social_net)
    modularity = community.modularity(communities, social_net)
    community_counts = Counter(communities.values())
    print('Community structure modularity: {:.3f}'.format(modularity))
    print('Number of communities: {}'.format(len(community_counts)))
    community_cumulative_plot(communities)
