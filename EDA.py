from settings import file_names
import matplotlib.pyplot as plt
import graph
import numpy as np
from collections import Counter
import powerlaw
import community
import igraph
import random
import json

random.seed(1)

def plot_powerlaw_fit(node_degrees, xlabel='Node degree', ylabel='Count'):
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


def louvain_community_cumulative_plot(communities: dict):
    community_counts = Counter(communities.values())
    community_counts = sorted(community_counts.values(), reverse=True)
    cumulative_community_counts = list(map(lambda x: x/sum(community_counts), np.cumsum(community_counts)))
    plt.plot(cumulative_community_counts)
    plt.title('Cumulative community membership for Louvain algorithm')
    plt.grid()
    plt.xlabel('Number of communities')
    plt.ylabel('Percentage of users represented')
    plt.show()


def run_louvain(social_net):
    print('Running Louvain algorithm for community detection...')
    communities = community.best_partition(social_net)
    modularity = community.modularity(communities, social_net)
    community_counts = Counter(communities.values())
    print('Community structure modularity: {:.3f}'.format(modularity))
    print('Number of communities: {}'.format(len(community_counts)))
    louvain_community_cumulative_plot(communities)
    return communities


def run_infomap():
    pass


def plot_walktrap_community_detection(social_net, n_clusters_list=None):
    walktrap = social_net.community_walktrap()

    for n_clusters in range(1000, 10000, 1000) if n_clusters_list is None else n_clusters_list:
        communities = walktrap.as_clustering(n_clusters)
        community_counts = Counter(communities.membership)
        community_counts = sorted(community_counts.values(), reverse=True)

        cumulative_community_counts = list(map(lambda x: x / sum(community_counts), np.cumsum(community_counts)))
        plt.plot(cumulative_community_counts, label=str(n_clusters))

    plt.title('Cumulative community membership for Walktrap algorithm')
    plt.grid()
    plt.xlabel('Number of communities')
    plt.ylabel('Percentage of users represented')
    plt.ylim(0, 1)
    plt.legend()
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
    communities = run_louvain(social_net)
    json.dump(communities, open(file_names['community_partition'], 'w+'))
