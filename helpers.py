import pandas as pd
from settings import file_names
import json
import numpy as np
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

    community_counts = Counter(communities.values())

    ratings = reviews_df.pivot_table(values='rating', aggfunc=np.mean, index='business_id',
                                                        columns='community')
    counts = reviews_df.pivot_table(values='rating', aggfunc=len, index='business_id',
                                                        columns='community')
    percentage_visited = counts.copy()
    for community in community_counts.keys():
        percentage_visited[community] = percentage_visited[community].apply(lambda count: count/community_counts[community])

    return ratings, counts, percentage_visited


