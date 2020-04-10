import pandas as pd
from settings import file_names
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.cluster import hierarchy

def split_train_validation_test(train_size=.7, validation_size=.15):
    assert train_size + validation_size < 1, 'Train and validation sizes must add up to less than 1'

    reviews = pd.read_csv(file_names['toronto_reviews_without_text'])
    reviews.date = pd.to_datetime(reviews.date)
    reviews = reviews.set_index('date').sort_index()

    train_validation_split = int(len(reviews.index) * train_size)

    if validation_size != 0:
        validation_test_split = int(len(reviews.index) * (train_size + validation_size))
    else:
        validation_test_split = train_validation_split + 1

    train_df = reviews.iloc[:train_validation_split]
    validation_df = reviews.iloc[train_validation_split:validation_test_split]
    test_df = reviews.iloc[validation_test_split:]

    return train_df, validation_df, test_df



def plot_dendrogram(G, partitions):

    num_of_nodes = G.number_of_nodes()
    dist = np.ones( shape=(num_of_nodes, num_of_nodes), dtype=np.float )*num_of_nodes
    d = num_of_nodes-1
    for partition in partitions:
        for subset in partition:
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    subsetl = list(subset)

                    dist[int(subsetl[i]), int(subsetl[j])] = d
                    dist[int(subsetl[j]), int(subsetl[i])] = d
        d -= 1



    dist_list = [dist[i,j] for i in range(num_of_nodes) for j in range(i+1, num_of_nodes)]


    Z = hierarchy.linkage(dist_list, 'complete')
    plt.figure()
    dn = hierarchy.dendrogram(Z)
