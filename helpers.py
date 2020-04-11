import pandas as pd
from settings import file_names
import json
import numpy as np
from scipy.cluster import hierarchy
from collections import defaultdict
from collections import Counter


def split_train_validation_test(train_size: float = .7, validation_size: float = .15):
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


def make_community_business_matrices(communities: dict = None, date_threshold='2018-10-10'):
    """
    Creates community-business interaction matrices
    :param communities: community split
    :param date_threshold: considers reviews before this date threshold
    :return: business mean rating per community, number of ratings per communities, percentage of visits per community
    """
    if communities is None:
        communities = json.load(open(file_names['community_partition']))

    reviews_df = pd.read_csv(file_names['toronto_reviews_without_text'])
    reviews_df.date = pd.to_datetime(reviews_df.date)

    reviews_df = reviews_df.set_index('date').loc[:date_threshold]
    reviews_df['community'] = reviews_df.user_id.apply(lambda user: communities[user])
def plot_dendrogram(G, partitions):
    '''Plot dendogram from output of Girven Newman community detection algorithm'''
    num_of_nodes = G.number_of_nodes()
    dist = np.ones( shape=(num_of_nodes, num_of_nodes), dtype=np.float )*num_of_nodes
    d = num_of_nodes-1
    for partition in partitions:
        for subset in partition:
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    subsetl = list(subset)

    community_counts = Counter(communities.values())

    ratings = reviews_df.pivot_table(values='rating', aggfunc=np.mean, index='business_id',
                                                        columns='community')
    counts = reviews_df.pivot_table(values='rating', aggfunc=len, index='business_id',
                                                        columns='community')
    percentage_visited = counts.copy()
    for community in community_counts.keys():
        percentage_visited[community] = percentage_visited[community].apply(lambda count: count/community_counts[community])

    return ratings, counts, percentage_visited



    Z = hierarchy.linkage(dist_list, 'complete')
    plt.figure()
    dn = hierarchy.dendrogram(Z)
    
    
    
def get_top_n(predictions, n=10):
    '''For the surprise recommender system library.
    Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
