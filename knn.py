import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split as train_test_split_2
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error
import warnings
import matplotlib.pyplot as plt

data_100k = 'ml-100k/u.data'

df = pd.read_csv(data_100k ,header=None, sep='\t',names=["user_id", "movie_id", "rating","timestamp"])
n_users, n_items = df['user_id'].unique().shape[0], df['movie_id'].unique().shape[0]
# user_ratings = user_ratings.sort_values('user_id').reset_index(drop = True)
df = df.drop(columns='timestamp')


# item based knn recommender
ratings = np.zeros((n_users, n_items))
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]

# user based knn recomender
items = np.zeros((n_items, n_users))
for row in df.itertuples():
    items[row[2]-1, row[1]-1] = row[3]

# print(len(items))
# print(len(items[0]))

# split dataset into train and test
def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    # print(ratings.shape[0])
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=10, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
    assert(np.all((train * test) == 0)) 
    return train, test


train, test = train_test_split(ratings)

train_item_based, test_item_based = train_test_split_2(items, test_size=0.33, random_state=42)

# calculate similarity
def fast_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors

    sim = ratings.dot(ratings.T) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)
 
user_similarity = fast_similarity(train, kind='user')
item_similarity = fast_similarity(train_item_based)

def predict_fast_simple(ratings, similarity, kind='user'):
    if kind == 'user':
#         print(np.array([np.abs(similarity).sum(axis=1)]).T)
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T


def get_rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.sqrt(mean_squared_error(pred, actual))

user_prediction = predict_fast_simple(train, user_similarity, kind='user')
item_prediction = predict_fast_simple(train_item_based, item_similarity)
# print(user_prediction)
print ('User-based CF MSE: ' + str(get_rmse(user_prediction, test)))
print ('Item-based CF MSE: ' + str(get_rmse(item_prediction, test_item_based)))



warnings.filterwarnings("ignore")
def predict_topk(ratings, similarity, kind='user', k=40):
    pred = np.zeros(tuple(ratings.shape))
    if kind == 'user':
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))       
    
    return pred
pred = predict_topk(train, user_similarity, kind='user', k=40)
pred_item = predict_topk(train_item_based, item_similarity, k=40)
print ('Top-k User-based CF MSE: ' + str(get_rmse(pred, test)))
print ('Top-k Item-based CF MSE: ' + str(get_rmse(pred_item, test_item_based)))

k_array = [5, 15, 30, 40, 50, 60, 100, 120]
user_train_rmse = []
user_test_rmse = []
item_train_rmse = []
item_test_rmse = []
for k in k_array:
    user_pred = predict_topk(train, user_similarity, kind='user', k=k)
    user_train_rmse += [get_rmse(user_pred, train)]
    user_test_rmse += [get_rmse(user_pred, test)]

    item_pred = predict_topk(train_item_based, item_similarity, k=k)
    item_train_rmse += [get_rmse(item_pred, train_item_based)]
    item_test_rmse += [get_rmse(item_pred, test_item_based)]





for i in range(len(k_array)):
  print(f'k={k_array[i]}, RMSE in user based test result={user_test_rmse[i]}')

for i in range(len(k_array)):
  print(f'k={k_array[i]}, RMSE in item based test result={item_test_rmse[i]}')

plt.plot(k_array, user_train_rmse, color='red', label='test result')
plt.plot(k_array, user_test_rmse, color='green', label='train result')
plt.show()

plt.plot(k_array, item_train_rmse, color='red', label='test result')
plt.plot(k_array, item_test_rmse, color='green', label='train result')
plt.show()


 #由上可知，最优的k落在k=40和k=60之间
k_array = []
for i in range(40,61):
  k_array.append(i)
user_train_rmse = []
user_test_rmse = []

item_train_rmse = []
item_test_rmse = []


# TODO 下面好像跑不动
for k in k_array:
    user_pred = predict_topk(train, user_similarity, kind='user', k=k)
    user_train_rmse += [get_rmse(user_pred, train)]
    user_test_rmse += [get_rmse(user_pred, test)]

for k in k_array:
    item_pred = predict_topk(train_item_based, item_similarity, k=k)
    item_train_rmse += [get_rmse(user_pred, train)]
    item_test_rmse += [get_rmse(user_pred, test)]


# show user based
for i in range(len(k_array)):
  print(f'k={k_array[i]}, RMSE in user based test result={user_test_rmse[i]}')


plt.plot(k_array, user_train_rmse, color='red', label='test result')
plt.plot(k_array, user_test_rmse, color='green', label='train result')
plt.show()

# show item based
for i in range(len(k_array)):
  print(f'k={k_array[i]}, RMSE in item based test result={item_test_rmse[i]}')


plt.plot(k_array, item_train_rmse, color='red', label='test result')
plt.plot(k_array, item_test_rmse, color='green', label='train result')
plt.show()