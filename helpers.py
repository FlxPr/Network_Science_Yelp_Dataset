import pandas as pd
from settings import file_names
import json
import numpy as np
from scipy.cluster import hierarchy
from collections import defaultdict
from collections import Counter
from matplotlib import pyplot as plt


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


def make_community_business_matrices(communities: dict = None, date_threshold: str = '2018-10-10'):
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

    community_counts = Counter(communities.values())

    mean_ratings = reviews_df.pivot_table(values='rating', aggfunc=np.mean, index='business_id',
                                                        columns='community')
    visit_counts = reviews_df.pivot_table(values='rating', aggfunc=len, index='business_id',
                                                        columns='community')
    visit_percentage = visit_counts.copy()
    for community in community_counts.keys():
        visit_percentage[community] = visit_percentage[community].apply(lambda count: count/community_counts[community])

    return mean_ratings, visit_counts, visit_percentage


def _assign_mean_rating(row, communities, user_column, business_column, min_community_size, min_community_visitors):
    pass


def compute_community_related_columns(df: pd.DataFrame, communities: dict=None, user_column: str = 'user_id',
                                         business_column: str = 'business_id', date_threshold: str= '2018-10-10',
                                         min_community_size: int = 10, min_community_visitors=30):
    if communities is None:
        communities = json.load(open(file_names['community_partition']))

    df = df.copy()
    ratings, counts, percentage_visited = make_community_business_matrices(communities, date_threshold)

    df['community_mean_rating'] = df.apply(lambda row: ratings.loc[row[business_column], communities[row[user_column]]],
                                           axis=1)
    df['community_percentage_of_visits'] = df.apply(lambda row: percentage_visited.loc[row[business_column], communities[row[user_column]]],
                                           axis=1)


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


def plot_dendrogram(G, partitions):
    num_of_nodes = G.number_of_nodes()
    dist = np.ones(shape=(num_of_nodes, num_of_nodes), dtype=np.float) * num_of_nodes
    d = num_of_nodes - 1
    for partition in partitions:
        for subset in partition:
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    subsetl = list(subset)

                    dist[int(subsetl[i]), int(subsetl[j])] = d
                    dist[int(subsetl[j]), int(subsetl[i])] = d
        d -= 1

    dist_list = [dist[i, j] for i in range(num_of_nodes) for j in range(i + 1, num_of_nodes)]
    Z = hierarchy.linkage(dist_list, 'complete')
    plt.figure()
    dn = hierarchy.dendrogram(Z)
