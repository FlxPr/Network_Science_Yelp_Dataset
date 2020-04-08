import pandas as pd
from tqdm import tqdm
import json
import igraph
from settings import file_names


df = pd.read_json(file_names['review'], encoding='latin')


def init_empty_lists(number_of_lists=4, list_length=100):
    for i in range(number_of_lists):
        yield [None] * list_length


def read_review_data(filtered_businesses):  # Todo implement filter here
    """
    Reads the review data and returns it as a pandas DataFrame
    The data file is 4GB. Might need to downsample it first (e.g. take only 1 country)
    :return:
    """
    line_count = len(open(file_names['review'], encoding='utf-8').readlines())
    user_ids, business_ids, stars, dates, texts = [], [], [], [], []
    with open(file_names['review'], encoding='utf-8') as f:
        for line in tqdm(f, total=line_count):
            blob = json.loads(line)
            if blob["business_id"] in filtered_businesses:
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
        'business_id': business_id,
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


def filter_friend_list(friend_list: str, filtered_friends_dict):
    friend_list = friend_list.split(", ")
    friend_list = filter(lambda x: filtered_friends_dict.get(x), friend_list)
    return ', '.join(friend_list)


def read_user_data(filtered_users_dict, parse_details=False):
    with open(file_names['user'], encoding='utf-8') as f:
        line_count = len(f.readlines())

    with open(file_names['user'], encoding='utf-8') as f:
        user_id, friends = [None] * line_count, [None] * line_count
        for i, line in enumerate(tqdm(f, total=line_count)):
            blob = json.loads(line)
            if filtered_users_dict.get(blob["user_id"]):
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

    df = pd.DataFrame({
        'user_id': user_id,
        'friends': friends
    }).dropna(subset=['user_id'])

    df.friends = df.friends.apply(lambda x: filter_friend_list(x, filtered_users_dict))

    return df


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
        if not i % 1000:
            print('Adding edges for user {}/{}'.format(i, len(ids)))
        graph.add_edges(((user_id, friend_id) for friend_id in friend_ids[i]))

    if save:
        graph.write_pickle(fname=file_names['user_graph'])

    return graph


if __name__ == '__main__':
    # Filter businesses in Toronto
    business_df = read_business_data()
    business_df = business_df[business_df.city == 'Toronto']
    business_df = business_df.dropna(subset=['categories'])
    business_df = business_df[business_df.categories.str.contains('Restaurant')]
    business_df.to_csv(file_names['toronto_businesses'], index=None)

    # Filter reviews in Toronto -- Very lengthy
    reviews_df = read_review_data(filtered_businesses=business_df.business_id.unique())
    reviews_df.to_csv(file_names['toronto_reviews'], index=None)

    # Remove users with only one review in Toronto -- Cuts by half the number of users, by 10% the review count
    review_count = reviews_df.user_id.value_counts()
    reviews_df.user_id = reviews_df.user_id.apply(lambda user_id: user_id if review_count[user_id] > 1 else None)
    reviews_df = reviews_df.dropna(subset=['user_id'])

    # Save light version of reviews as csv -- to be uploaded on GitHub
    reviews_df.drop('text', axis=1).to_csv(file_names['toronto_reviews_without_text'])

    # Filter users active in Toronto
    reviews_df = pd.read_csv(file_names['toronto_reviews_without_text'])
    toronto_locals = dict.fromkeys(reviews_df.user_id.unique(), 1)
    user_df = read_user_data(filtered_users_dict=toronto_locals)
    user_df.to_csv(file_names['toronto_users'], index=None)
