import pandas as pd
from settings import file_names

from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score


df = pd.read_csv(file_names['toronto_reviews_without_text'])
print('Number of reviews:', len(df))

data = Dataset()
data.fit(df['user_id'], df['business_id'])
_, M = data.build_interactions(df[['user_id', 'business_id', 'rating']].to_records(index=False).tolist())

# Split the data into a training and test set
M_train, M_test = random_train_test_split(M, test_percentage=0.2)

# Initialize our model
model = LightFM(no_components=20, loss='warp')

# Training the model on the training set
model.fit(M_train, epochs=50, num_threads=16)

# Calculate AUC metrics
train_auc = auc_score(model, M_train, num_threads=16).mean()
test_auc = auc_score(model, M_test, num_threads=16).mean()

print('Train AUC:', train_auc)
print('Test AUC:', test_auc)
