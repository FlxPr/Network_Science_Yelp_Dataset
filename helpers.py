import pandas as pd
from settings import file_names, split_dates
import json
import numpy as np
from scipy.cluster import hierarchy
from collections import defaultdict, Counter
from matplotlib import pyplot as plt
from functools import reduce


def split_train_validation_test(reviews_df):
    reviews_df = reviews_df.copy()
    reviews_df.date = pd.to_datetime(reviews_df.date)
    reviews_df = reviews_df.set_index('date').sort_index()

    train_df = reviews_df.loc[:split_dates['train']['end']]
    validation_df = reviews_df.iloc[split_dates['validation']['begin']:split_dates['validation']['end']]
    test_df = reviews_df.iloc[split_dates['test']['begin']:split_dates['test']['end']]

    return train_df, validation_df, test_df


def make_community_business_matrices(reviews_df: pd.DataFrame, communities: dict = None,
                                     date_threshold: str = '2018-10-10'):
    """
    Creates community-business interaction matrices
    :param communities: single community split
    :param date_threshold: only considers reviews before this date threshold
    :return: business mean rating per community, number of ratings per communities, percentage of visits per community
    """
    if communities is None:
        communities = json.load(open(file_names['community_partition']))

    reviews_df = reviews_df.copy()
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

    visit_counts['all_dataset'] = visit_counts.apply(lambda row: np.sum(row), axis=1)
    visit_percentage['all_dataset'] = visit_counts['all_dataset']/np.sum(list(community_counts.values()))
    mean_ratings['all_dataset'] = reduce(pd.Series.add, [visit_counts[community].fillna(0) *
                                                         mean_ratings[community].fillna(0)
                                                         for community in community_counts.keys()])\
                                  / visit_counts['all_dataset']

    return mean_ratings, visit_counts, visit_percentage


def compute_community_related_features(reviews_df: pd.DataFrame, communities: list = None, user_column: str = 'user_id',
                                       business_column: str = 'business_id', date_threshold: str = split_dates['train']['end'],
                                       min_community_size: int = 10, min_community_visitors=10):
    """
    Computes statistics for every review based on the reviewer's community
    :param reviews_df: reviews dataframe
    :param communities: community splits, list of dictionary-like objects in the format {user_id: community_id}
    :param user_column: name of the user_id column in df
    :param business_column: name of the business_id column in df
    :param date_threshold: considers reviews only before that threshold
    :param min_community_size: minimum member count in community to compute statistics. Communities with less than this
     member count will be assigned statistics computed on the whole dataset
    :param min_community_visitors: minimum of reviews for a given restaurant per community to compute community
    statistics. Reviews with less than this threshold will be assigned statistics computed on the whole dataset.
    :return: review_df with new columns, for each community split:
    community of the user giving the review
    mean rating of the user's community for the reviewed restaurant
    percentage of the user's community members that gave a review to this restaurant
    The DataFrame will contain NaN values for restaurants that have no review before the date threshold
    """
    reviews_df = reviews_df.copy()

    if communities is None:  # Use default community split
        communities = [json.load(open(file_names['community_partition']))]

    for i, community in enumerate(communities):
        ratings, counts, percentage_visited = make_community_business_matrices(reviews_df, community, date_threshold)

        filtered_businesses = {business_id: 1 for business_id in ratings.index}
        community_counts = Counter(community.values())
        filtered_communities = {community: 1 for community in community_counts.keys()
                                if community_counts[community] >= min_community_size}
        reviews_df['community_{}'.format(i)] = reviews_df[user_column].apply(lambda user_id: community[user_id])
        reviews_df['community_{}_mean_rating'.format(i)] = reviews_df.apply(
            lambda review: None if not filtered_businesses.get(review[business_column])
            else(
                ratings.loc[review[business_column], review['community_{}'.format(i)]]
                if filtered_communities.get(review['community_{}'.format(i)])
                and counts.loc[review[business_column], review['community_{}'.format(i)]] >= min_community_visitors
                else (ratings.loc[review[business_column], 'all_dataset'])
            ), axis=1
        )
        reviews_df['community_{}_percentage_of_visits'.format(i)] = reviews_df.apply(
            lambda review: None if not filtered_businesses.get(review[business_column])
            else (
                percentage_visited.loc[review[business_column], review['community_{}'.format(i)]]
                if filtered_communities.get(review['community_{}'.format(i)])
                   and counts.loc[review[business_column], review['community_{}'.format(i)]] >= min_community_visitors
                else percentage_visited.loc[review[business_column], 'all_dataset']
            ), axis=1
        )

    return reviews_df


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


if __name__ == '__main__':
    df = pd.read_csv(file_names['toronto_reviews_without_text'])
    df = compute_community_related_features(df)
    print('Result of the computing of community-based features:')
    print(df.head(5))
