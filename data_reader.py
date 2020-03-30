import pandas as pd
from tqdm import tqdm
import json
import numpy as np
import igraph
from settings import file_names


def init_empty_lists(number_of_lists=4, list_length=100):
    for i in range(number_of_lists):
        yield [None] * list_length


def read_review_data():  # Todo implement filter here
    """
    Reads the review data and returns it as a pandas DataFrame
    The data file is 4GB. Might need to downsample it first (e.g. take only 1 country)
    :return:
    """
    line_count = len(open(file_names['review'], encoding='utf-8').readlines())
    user_ids, business_ids, stars, dates, texts = [], [], [], [], []
    with open("review.json", encoding='utf-8') as f:
        for line in tqdm(f, total=line_count):
            blob = json.loads(line)
            user_ids += [blob["user_id"]]
            business_ids += [blob["business_id"]]
            stars += [blob["stars"]]
            dates += [blob["date"]]
            texts += [blob["text"]]
    ratings = pd.DataFrame(
        {"user_id": user_ids,
         "business_id": business_ids,
         "rating": stars,
         "date": dates,
         "text": texts}
    )
    return ratings


def read_checkin_data():
    pass


def read_tip_data():
    pass


def read_business_data(filter_by_city=None, save_to_csv=False):
    with open(file_names['business'], encoding='utf-8') as f:
        line_count = len(f.readlines())

    business_id, name, address, city = init_empty_lists(4, line_count)
    state, postal_code, latitude, longitude = init_empty_lists(4, line_count)
    stars, review_count, attributes, categories = init_empty_lists(4, line_count)

    with open(file_names['business'], encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=line_count)):
            blob = json.loads(line)
            business_id[i] = blob["business_id"]
            name[i] = blob["name"]
            address[i] = blob["address"]
            city[i] = blob["city"]
            state[i] = blob["state"]
            postal_code[i] = blob["postal_code"]
            latitude[i] = blob["latitude"]
            longitude[i] = blob["longitude"]
            stars[i] = blob["stars"]
            review_count[i] = blob["review_count"]
            attributes[i] = blob["attributes"]
            categories[i] = blob["categories"]

    df = pd.DataFrame({
        'business_id':business_id,
        'name': name,
        'address': address,
        'city': city,
        'state': state,
        'postal_code': postal_code,
        'latitude': latitude,
        'longitude': longitude,
        'stars': stars,
        'review_count': review_count,
        'attributes': attributes,
        'categories': categories
    })

    if save_to_csv:
        df.to_csv(file_names['business_csv'])

    return df


def read_user_data(parse_details=False, nrows=np.inf):
    with open(file_names['user'], encoding='utf-8') as f:
        line_count = len(f.readlines())

    with open(file_names['user'], encoding='utf-8') as f:
        line_count = min(nrows, line_count)
        user_id, friends = [None] * line_count, [None] * line_count
        for i, line in enumerate(tqdm(f, total=line_count)):
            if i == nrows:
                break
            blob = json.loads(line)
            user_id[i] = blob["user_id"]
            friends[i] = blob["friends"]
            if parse_details:
                pass  # TODO if useful, parse details about users:
                # user_name += [blob["name"]]
                # review_count += [blob["review_count"]]
                # yelping_since += [blob["yelping_since"]]
                # useful += [blob["yelping_since"]]
                # funny += [blob["yelping_since"]]
                # cool += [blob["text"]]
                # elite += [blob["text"]]

    return user_id, friends


def read_photo_data():
    pass


def make_friend_graph(save=True):
    """
    Currently does not run due to performance issues.
    :param save:
    :return:
    """
    ids, friend_ids = read_user_data()
    for i in range(len(friend_ids)):
        friend_ids[i] = filter(lambda friend_id: friend_id in ids, friend_ids[i].split(', '))

    graph = igraph.Graph()
    graph.add_vertices(ids)
    for i, user_id in enumerate(ids):
        if not i%1000:
            print('Adding edges for user {}/{}'.format(i, len(ids)))
        graph.add_edges(((user_id, friend_id) for friend_id in friend_ids[i]))


    if save:
        graph.write_pickle(fname=file_names['user_graph'])
    return graph

