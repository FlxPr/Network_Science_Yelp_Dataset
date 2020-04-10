import pandas as pd
from settings import file_names


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
