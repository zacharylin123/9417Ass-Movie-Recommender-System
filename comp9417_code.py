#!/usr/bin/env python

import numpy as np
import pandas as pd
import math
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp
import warnings
from sklearn.metrics import roc_auc_score
import random
from collections import defaultdict
from matplotlib import pyplot as plt


data_100k = 'ml-100k/u.data'
warnings.filterwarnings("ignore")

def prepare_data(test_size=0.2, datafile='ml-100k/u.data', header=['uid','iid','ratings','timestamp'], sep='\t', seed=0):
	# Read CSV File into A Pandas DataFrame
	df = pd.read_csv(datafile, header=None, names=header, sep=sep, engine='python')
	df.drop(columns='timestamp')
	# print(df.head())
	# The Number of User and Items
	num_users, num_items = df[header[0]].unique().shape[0], df[header[1]].unique().shape[0]
	# The minimum id of user and item (because in Python array index is from 0)
	uid_min, iid_min = df['uid'].min(), df['iid'].min()

	# Train and Test Dataset Splitting
	train_df, test_df = train_test_split(np.asarray(df), test_size=test_size, random_state=seed)

	# Change the data structure into sparse matrix
	train = sp.csr_matrix((train_df[:, 2], (train_df[:, 0]-uid_min, train_df[:, 1]-iid_min)), shape=(num_users, num_items))
	test = sp.csr_matrix((test_df[:, 2], (test_df[:, 0]-uid_min, test_df[:, 1]-iid_min)), shape=(num_users, num_items))

	print("Number of Users: " + str(num_users))
	print("Number of Items: " + str(num_items))
	print("=" * 120)

	return train, test, num_users, num_items


class KNN(object):
    def __init__(self, train_matrix, test_matrix):
        self.train_matrix = train_matrix.toarray()
        self.test_matrix = test_matrix.toarray()

    def get_similarity_cos(self, ratings, epsilon=1e-9):
        sim = ratings.dot(ratings.T) + epsilon
        norms = np.array([np.sqrt(np.diagonal(sim))])
        return (sim / norms / norms.T)
        
    def get_similarity_euc(self, ratings):
        dist_sq = np.sum((ratings[:, np.newaxis, :] - ratings[np.newaxis, :, :]) ** 2, axis = -1)
        return dist_sq

    def predict_topk(self, ratings, similarity, k=40):
        pred = np.zeros(tuple(ratings.shape))
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:, i])[:-k-1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(
                    ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
        return pred

    def get_rmse(self, pred, actual):
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return np.sqrt(mean_squared_error(pred, actual))

    def predict_by_array(self, k_array=[5, 10, 15]):
        self.user_train_rmse = []
        self.user_test_rmse = []
        self.k_array = k_array
        # user_similarity = self.get_similarity_cos(ratings=self.train_matrix)
        user_similarity = self.get_similarity_euc(ratings=self.train_matrix)
        for k in k_array:
            user_pred = self.predict_topk(
                self.train_matrix, user_similarity, k=k)
            self.user_train_rmse += [
                self.get_rmse(user_pred, self.train_matrix)]
            self.user_test_rmse += [self.get_rmse(user_pred, self.test_matrix)]

    def plot(self):
        for i in range(len(self.k_array)):
            print(
                f'k={self.k_array[i]}, RMSE in test result={self.user_test_rmse[i]}')
        l1, = plt.plot(self.k_array, self.user_train_rmse,
                 color='red', label='test result')
        l2, = plt.plot(self.k_array, self.user_test_rmse,
                 color='green', label='train result')
        plt.legend(loc='best')
        plt.xlabel('num of training')
        plt.ylabel('RMSE')
        plt.title('RMSE of training and testing')
        plt.show()

class SVD(object):
    def __init__(self,train_matrix,test_matrix):
        # Initialize Parameters
        self.num_factors = 600 # Dimension of the Latent Factor
        self.regs = 1e-3 # Regularizer Coefficient
        self.lr = 0.01 # Learning Rate
        self.trains = 50 # How many number of Training Loops
        self.batch_size = 228 # How many data is fed into the training algorithm in each training
        self.num_user, self.num_item = train_matrix.shape[0], train_matrix.shape[1]

        # Store the user IDs in a list, the item IDs in a list and the ratings in a list
        train_matrix, test_matrix = train_matrix.tocoo(), test_matrix.tocoo()
        self.train_uid, self.train_iid, self.train_ratings = list(train_matrix.row),list(train_matrix.col),list(train_matrix.data)
        self.test_uid, self.test_iid, self.test_ratings = list(test_matrix.row),list(test_matrix.col),list(test_matrix.data)

        # Calculate the average of all ratings (the mu value in the equation)
        self.mu = np.mean(self.train_ratings)

        # Total number of training data instances
        self.num_training = len(self.train_ratings)

        # Number of batches
        self.num_batch = int(self.num_training / self.batch_size)
        print("Data Preparation Completed.")

    # Build the model for customized SGD algorithm
        # Initialize all the parameters (Use Normal Distribution)
        # bu and bi are vectors (Note the dimension)
        self.bu = np.random.normal(scale = 1. / self.num_factors, size=[self.num_user])
        self.bi = np.random.normal(scale = 1. / self.num_factors, size=[self.num_item])

        # P and Q are matrices (Note the dimension)
        self.P = np.random.normal(scale=1. / self.num_factors, size=[self.num_user, self.num_factors])
        self.Q = np.random.normal(scale=1. / self.num_factors, size=[self.num_factors, self.num_item])
        print("Parameter Initialization Completed.")

    # Training using SGD algorithm
    def train_and_evaluate(self):
        self.train_result = []
        self.test_result = []
        for tr in range(self.trains):
            for uid, iid, ratings in list(zip(self.train_uid, self.train_iid, self.train_ratings)):
                # The estimated rating
                pred_r = self.mu + self.bu[uid] + self.bi[iid] + np.dot(self.Q[:, iid], self.P[uid, :])

                # Calculate the loss of this specific user-item pair
                error = ratings - pred_r

                # Update the parameters
                self.bu[uid] = self.bu[uid] + self.lr * (error - self.regs * self.bu[uid])
                self.bi[iid] = self.bi[iid] + self.lr * (error - self.regs * self.bi[iid])
                self.P[uid, :] = self.P[uid, :] + self.lr * (error * self.Q[:, iid] - self.regs * self.P[uid, :])
                self.Q[:, iid] = self.Q[:, iid] + self.lr * (error * self.P[uid, :] - self.regs * self.Q[:, iid])
            rms_test_list, rms_train_list = [], []
            
            for test in range(len(self.test_uid)):
                uid = self.test_uid[test]
                iid = self.test_iid[test]
                ratings = self.test_ratings[test]
                rms_test_list.append((self.mu + self.bu[uid] + self.bi[iid] + np.dot(self.Q[:, iid], self.P[uid, :]) - ratings) ** 2)
                
            for train in range(len(self.train_uid)):
                uid = self.train_uid[train]
                iid = self.train_iid[train]
                ratings = self.train_ratings[train]
                rms_train_list.append((self.mu + self.bu[uid] + self.bi[iid] + np.dot(self.Q[:, iid], self.P[uid, :]) - ratings) ** 2)
    
            rms_test = np.sqrt(np.mean(rms_test_list))
            self.test_result.append(rms_test)
            rms_train = np.sqrt(np.mean(rms_train_list))
            self.train_result.append(rms_train)
            print("The {0} Training: [RMS] {1} and Testing: [RMS] {2}".format(tr, rms_train, rms_test))
    def plot(self):
        x_axis = range(1,self.trains+1)
        plt.plot(x_axis, self.test_result, color='red', label='test result')
        plt.plot(x_axis, self.train_result, color='green', label='train result')
        plt.legend(loc='best')
        plt.xlabel('num of training')
        plt.ylabel('RMSE')
        plt.title('RMSE of training and testing')
        plt.show()

class BPR(object):
    def __init__(self,num_users, num_items):
        self.num_factors = 600 # Dimension of the Latent Factor
        self.reg = 0.01 # Regularizer Coefficient
        self.lr = 0.05 # Learning Rate
        self.train_count = 300 # How many number of Training Loops
        self.latent_factors = 20
        self.user_count, self.item_count = num_users, num_items
        self.train_data_path = 'ml-100k/u1.base'
        self.test_data_path = 'ml-100k/u1.test'
        self.size_u_i = self.user_count * self.item_count
        # latent_factors of U & V
        self.U = np.random.rand(self.user_count, self.latent_factors) * 0.01
        self.V = np.random.rand(self.item_count, self.latent_factors) * 0.01
        self.test_data = np.zeros((self.user_count, self.item_count))
        self.test = np.zeros(self.size_u_i)
        self.predict_ = np.zeros(self.size_u_i)

        print("Data Preparation Completed.")


    def load_data(self,path):
        user_ratings = defaultdict(set)
        max_u_id = -1
        max_i_id = -1
        with open(path, 'r') as f:
            for line in f.readlines():
                u, i, r,t = line.split("\t")
                u = int(u)
                i = int(i)
                user_ratings[u].add(i)
                max_u_id = max(u, max_u_id)
                max_i_id = max(i, max_i_id)
        return user_ratings

    def load_test_data(self, path):
        file = open(path, 'r')
        for line in file:
            line = line.split('\t')
            user = int(line[0])
            item = int(line[1])
            self.test_data[user - 1][item - 1] = 1

    def train(self, user_ratings_train):
        for user in range(self.user_count):
            # sample a user
            u = random.randint(1, self.user_count)
            if u not in user_ratings_train.keys():
                continue
            # sample a positive item from the observed items
            i = random.sample(user_ratings_train[u], 1)[0]
            # sample a negative item from the unobserved items
            j = random.randint(1, self.item_count)
            while j in user_ratings_train[u]:
                j = random.randint(1, self.item_count)
            u -= 1
            i -= 1
            j -= 1
            r_ui = np.dot(self.U[u], self.V[i].T)
            r_uj = np.dot(self.U[u], self.V[j].T)
            r_uij = r_ui - r_uj
            mid = 1.0 / (1 + np.exp(r_uij))
            temp = self.U[u]
            self.U[u] += -self.lr * (-mid * (self.V[i] - self.V[j]) + self.reg * self.U[u])
            self.V[i] += -self.lr * (-mid * temp + self.reg * self.V[i])
            self.V[j] += -self.lr * (-mid * (-temp) + self.reg * self.V[j])

    def predict(self, user, item):
        predict = np.mat(user) * np.mat(item.T)
        return predict

    def pre_handel(self, set, predict, item_count):
    # Ensure the recommendation cannot be positive items in the training set.
        for u in set.keys():
            for j in set[u]:
                predict[(u-1) * item_count + j - 1]= 0
        return predict

    def main(self):
        user_ratings_train = self.load_data(self.train_data_path)
        self.load_test_data(self.test_data_path)
        for u in range(self.user_count):
            for item in range(self.item_count):
                if int(self.test_data[u][item]) == 1:
                    self.test[u * self.item_count + item] = 1
                else:
                    self.test[u * self.item_count + item] = 0
        for i in range(self.user_count * self.item_count):
            self.test[i] = int(self.test[i])
        # training
        auc_list = []
        for i in range(self.train_count):
            self.train(user_ratings_train)
            predict_matrix = self.predict(self.U, self.V)
            # prediction
            self.predict_ = predict_matrix.getA().reshape(-1)
            self.predict_ = self.pre_handel(user_ratings_train, self.predict_, self.item_count)
            auc_score = roc_auc_score(self.test, self.predict_)
            auc_list.append(auc_score)
            print('AUC:', auc_score)
        plt.plot(range(1,self.train_count+1), auc_list,
                 color='green', label='AUC score')
        plt.legend(loc='best')
        plt.xlabel('num of training')
        plt.ylabel('AUC')
        plt.title('AUC trend with increasing training times')
        plt.show()

mode = 'knn'
if mode == 'knn':
    train, test, num_users, num_items = prepare_data()
    knn = KNN(train,test)
    knn.predict_by_array(k_array = range(5,50,5))
    knn.plot()
elif mode == 'svd':
    train, test, num_users, num_items = prepare_data()
    svd = SVD(train,test)
    svd.train_and_evaluate()
    svd.plot()
elif mode == 'bpr':
    train, test, num_users, num_items = prepare_data()
    bpr = BPR(num_users, num_items)
    bpr.main()