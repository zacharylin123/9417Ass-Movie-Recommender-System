# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:59:25 2017

@author: gzs10228
"""

import re
import datetime
import pandas as pd
from pandas import DataFrame
from numpy import *
from numpy import linalg as la
import numpy as np
import os

'''以下是三种计算相似度的算法，分别是欧式距离、皮尔逊相关系数和余弦相似度,
注意三种计算方式的参数inA和inB都是列向量'''

def ecludSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))  # 范数的计算方法linalg.norm()，这里的1/(1+距离)表示将相似度的范围放在0与1之间

def pearsSim(inA, inB):
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]  # 皮尔逊相关系数的计算方法corrcoef()，参数rowvar=0表示对列求相似度，这里的0.5+0.5*corrcoef()是为了将范围归一化放到0和1之间

def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)  # 将相似度归一s到0与1之间 

def standEst(dataMat,user,cosSim,item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: 
            continue
        overLap = np.nonzero(np.logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        if len(overLap) == 0: 
            similarity = 0
        else: 
            similarity = cosSim(dataMat[overLap,item],dataMat[overLap,j])
        print('the item%d and item%d similarity is：%f' %(item,j,similarity))
        simTotal = simTotal + similarity
        ratSimTotal = ratSimTotal + similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal
    
'''按照前k个奇异值的平方和占总奇异值的平方和的百分比percentage来确定k的值,
后续计算SVD时需要将原始矩阵转换到k维空间'''
def sigmaPct(sigma, percentage):
    sigma2 = sigma ** 2  # 对sigma求平方
    sumsgm2 = sum(sigma2)  # 求所有奇异值sigma的平方和
    sumsgm3 = 0  # sumsgm3是前k个奇异值的平方和
    k = 0
    for i in sigma:
        sumsgm3 += i ** 2
        k += 1
        if sumsgm3 >= sumsgm2 * percentage:
            return k


'''函数svdEst()的参数包含：数据矩阵、用户编号、物品编号和奇异值占比的阈值，
数据矩阵的行对应用户，列对应物品，函数的作用是基于item的相似性对用户未评过分的物品进行预测评分'''
    
def svdEst(dataMat, user, simMeas, item, percentage):
    n = np.shape(dataMat)[1]
    simTotal = 0.0;
    ratSimTotal = 0.0
    u, sigma, vt = la.svd(dataMat)
    k = sigmaPct(sigma, percentage)  # 确定k的值
    sigmaK = np.mat(np.eye(k) * sigma[:k])  # 构建对角矩阵
    xformedItems = dataMat.T * u[:, :k] * sigmaK.I  # 根据k的值将原始数据转换到k维空间(低维),xformedItems表示物品(item)在k维空间转换后的值
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item: continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)  # 计算物品item与物品j之间的相似度
        simTotal += similarity  # 对所有相似度求和
        ratSimTotal += similarity * userRating  # 用"物品item和物品j的相似度"乘以"用户对物品j的评分"，并求和
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal  # 得到对物品item的预测评分

'''生成离线特想向量之间的相似度'''

def offline_similarity(dataMat,cosSim,percentage):
    n = np.shape(dataMat)[1]
    offline_similarity = []
    offline_similarity = np.eye(n)
#    for i in range(n):
#        one_array = np.zeros(n)
#        offline_similarity.append(one_array)
    u, sigma, vt = la.svd(dataMat)
    k = sigmaPct(sigma, percentage)  # 确定k的值
    sigmaK = np.mat(np.eye(k) * sigma[:k])  # 构建对角矩阵
    xformedItems = dataMat.T * u[:, :k] * sigmaK.I  # 根据k的值将原始数据转换到k维空间(低维),xformedItems表示物品(item)在k维空间转换后的值
    '''存储在上三角'''
    for i in range(n-1):
        for j in range(i+1,n):
#            print a[i][j]
            offline_similarity[i][j] = cosSim(xformedItems[i, :].T, xformedItems[j, :].T) 
            offline_similarity[j][i] = offline_similarity[i][j]
#            offline_similarity[j-k][k] = offline_similarity[k][j-k]
    offline_mat = np.mat(offline_similarity)
#    with open('offline_similarity.txt','w') as f:
#        f.write(offline_similarity)
    return offline_mat,offline_similarity

def svdEst_offline(dataMat, user, cosMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0;
    ratSimTotal = 0.0
#    u, sigma, vt = la.svd(dataMat)
#    k = sigmaPct(sigma, percentage)  # 确定k的值
#    sigmaK = np.mat(np.eye(k) * sigma[:k])  # 构建对角矩阵
#    xformedItems = dataMat.T * u[:, :k] * sigmaK.I  # 根据k的值将原始数据转换到k维空间(低维),xformedItems表示物品(item)在k维空间转换后的值
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item: continue
#        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)  # 计算物品item与物品j之间的相似度
        similarity = offline_similarity[item][j]  # 计算物品item与物品j之间的相似度
        simTotal += similarity  # 对所有相似度求和
        ratSimTotal += similarity * userRating  # 用"物品item和物品j的相似度"乘以"用户对物品j的评分"，并求和
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal  # 得到对物品item的预测评分

    
'''函数recommend()产生预测评分最高的N个推荐结果，默认返回3个；
参数包括：数据矩阵、用户编号、相似度衡量的方法、预测评分的方法、以及奇异值占比的阈值；
数据矩阵的行对应用户，列对应物品，函数的作用是基于item的相似性对用户未评过分的物品进行预测评分；
相似度衡量的方法默认用余弦相似度'''

def recommend(dataMat, user, N, cosSim, svdEst_offline):
    N=3
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]  # 建立一个用户未评分item的列表
    if len(unratedItems) == 0: return 'you rated everything'  # 如果都已经评过分，则退出
    itemScores = []
    for item in unratedItems:  # 对于每个未评分的item，都计算其预测评分
        estimatedScore = svdEst_offline(dataMat, user, cosSim, item)
        print('%d,%f'%(item,estimatedScore))
        itemScores.append((item, estimatedScore))
    itemScores = sorted(itemScores, key=lambda x: x[1], reverse=True)  # 按照item的得分进行从大到小排序
    # print (itemScores)
    return itemScores[:N]  # 返回前N大评分值的item名，及其预测评分值
    
def loadMovieLensTrain(fileName='u1.base'):
    str1 = './ml-100k/'                         # 目录的相对地址
    
    prefer = {}
    for line in open(str1+fileName,'r'):       # 打开指定文件
        (userid, movieid, rating,ts) = line.split('\t')     # 数据集中每行有4项
        prefer.setdefault(userid, {})      # 设置字典的默认格式,元素是user:{}字典
        prefer[userid][movieid] = float(rating)    

    return prefer      # 格式如{'user1':{itemid:rating, itemid2:rating, ,,}, {,,,}}
##==================================
#        加载对应的测试集文件
#  参数fileName 代表某个测试集文件,如u1.test
##==================================
def loadMovieLensTest(fileName='u1.test'):
    str1 = './ml-100k/'    
    prefer = {}
    for line in open(str1+fileName,'r'):    
        (userid, movieid, rating,ts) = line.split('\t')   #数据集中每行有4项
        prefer.setdefault(userid, {})    
        prefer[userid][movieid] = float(rating)   
    return prefer      

def get_dataMat():
    item_set = set()
    #item_set.add()
    for key in train_data.keys():
        for item in train_data[str(key)].keys():
            item_set.add(item)
    
    item_list = list(item_set)
    urs_list = list(set(train_data.keys()))
    
    data_matrix = []
    for i in range(len(urs_list)):
        one_array = np.zeros(len(item_list))
        data_matrix.append(one_array)   
    for dict_urs in train_data.items():
        urs_id = int(urs_list.index(dict_urs[0]))
    #    print urs_id
        for dict_item in dict_urs[1].items():
            item_id = int(item_list.index(dict_item[0]))
    #        print item_id
            score = dict_item[1]
    #        print score
            try:
                data_matrix[urs_id][item_id] = score
            except Exception:
                print('error') #str(e)   
    dataMat = np.mat(data_matrix)
    return dataMat,urs_list,item_list


if __name__ == '__main__':
#    pass
    train_data = loadMovieLensTrain()
    test_data = loadMovieLensTest()
#    time_1 = datetime.datetime.now()
    dataMat,urs_list,item_list = get_dataMat()
    offline_mat,offline_similarity = offline_similarity(dataMat,cosSim,percentage=0.9)
    itemScores = recommend(dataMat, user,N, cosSim, svdEst_offline)
#    time_2 = datetime.datetime.now()
#    run_time = time_2-time_1
#    print run_time
























