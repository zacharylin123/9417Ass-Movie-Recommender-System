#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Peidong
# @Site    : 
# @File    : MFRecommendSystem.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt

class SVD:
    def __init__(self,mat,K=20):
        self.mat=mat
        self.K=K
        self.bi={}
        self.bu={}
        self.qi={}
        self.pu={}
        self.avg=np.mean(self.mat[:,2])
        for i in range(self.mat.shape[0]):
            uid=self.mat[i,0]
            iid=self.mat[i,1]
            self.bi.setdefault(iid,0)
            self.bu.setdefault(uid,0)
            self.qi.setdefault(iid,np.random.random((self.K,1))/10*np.sqrt(self.K))
            self.pu.setdefault(uid,np.random.random((self.K,1))/10*np.sqrt(self.K))
    def predict(self,uid,iid):  #预测评分的函数
        #setdefault的作用是当该用户或者物品未出现过时，新建它的bi,bu,qi,pu，并设置初始值为0
        self.bi.setdefault(iid,0)
        self.bu.setdefault(uid,0)
        self.qi.setdefault(iid,np.zeros((self.K,1)))
        self.pu.setdefault(uid,np.zeros((self.K,1)))
        rating=self.avg+self.bi[iid]+self.bu[uid]+np.sum(self.qi[iid]*self.pu[uid]) #预测评分公式
        #由于评分范围在1到5，所以当分数大于5或小于1时，返回5,1.
        if rating>5:
            rating=5
        if rating<1:
            rating=1
        return rating
    
    def train(self,steps=30,gamma=0.04,Lambda=0.15):    #训练函数，step为迭代次数。
        print('train data size',self.mat.shape)
        for step in range(steps):
            print('step',step+1,'is running')
            KK=np.random.permutation(self.mat.shape[0]) #随机梯度下降算法，kk为对矩阵进行随机洗牌
            rmse=0.0
            for i in range(self.mat.shape[0]):
                j=KK[i]
                uid=self.mat[j,0]
                iid=self.mat[j,1]
                rating=self.mat[j,2]
                eui=rating-self.predict(uid, iid)
                rmse+=eui**2
                self.bu[uid]+=gamma*(eui-Lambda*self.bu[uid])  
                self.bi[iid]+=gamma*(eui-Lambda*self.bi[iid])
                tmp=self.qi[iid]
                self.qi[iid]+=gamma*(eui*self.pu[uid]-Lambda*self.qi[iid])
                self.pu[uid]+=gamma*(eui*tmp-Lambda*self.pu[uid])
            gamma=0.93*gamma
            print('rmse is',np.sqrt(rmse/self.mat.shape[0]))
    
    def test(self,test_data):  #gamma以0.93的学习率递减
        
        test_data=test_data
        print('test data size',test_data.shape)
        rmse=0.0
        for i in range(test_data.shape[0]):
            uid=test_data[i,0]
            iid=test_data[i,1]
            rating=test_data[i,2]
            eui=rating-self.predict(uid, iid)
            rmse+=eui**2
        print('rmse of test data is',np.sqrt(rmse/test_data.shape[0]))
            
            
def getData():   #获取训练集和测试集的函数
    import re
    f=open('C:/Users/xuwei/Desktop/data.txt','r')
    lines=f.readlines()
    f.close()
    data=[]
    for line in lines:
        list=re.split('\t|\n',line)
        if int(list[2]) !=0:    #提出评分0的数据，这部分是用户评论了但是没有评分的
            data.append([int(i) for i in list[:3]])
    random.shuffle(data)
    train_data=data[:int(len(data)*7/10)]
    test_data=data[int(len(data)*7/10):]
    print('load data finished')
    print('total data ',len(data))
    return train_data,test_data
    

# 读取u.data文件
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)

# 计算唯一用户和电影的数量
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

train_data, test_data = train_test_split(df, test_size=0.2, random_state = 0)

# 计算数据集的稀疏度
# sparsity = round(1.0 - len(df)/float(n_users*n_items), 3)
# print('The sparsity level of MovieLens100K is ' + str(sparsity*100) + '%')

# 创建uesr-item矩阵，此处需创建训练和测试两个UI矩阵
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]


# 使用SVD进行矩阵分解
# u, s, vt = svds(train_data_matrix,k = 7)
# s_diag_matrix = np.diag(s)
# X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

a=SVD(train_data_matrix,30)  
a.train()
a.test(test_data_matrix)

# 利用均方根误差进行评估


# def rmse(prediction, ground_truth):
#     prediction = prediction[ground_truth.nonzero()].flatten()
#     ground_truth = ground_truth[ground_truth.nonzero()].flatten()
#     return sqrt(mean_squared_error(prediction, ground_truth))

# print('User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))


