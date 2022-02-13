
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.stats as stats


import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

from tqdm import tqdm

from utils import *



def get_True_metric(model, dataset, k, metric):
    
    N = nums[dataset]

    _, Ru =  load_True(model, dataset)

    M = len(Ru)# total number of users

    True_PR = rank_order(Ru, N)/M

    metric_true = eval(metric + '_K(True_PR, k)')
    
    return metric_true[-1]


def get_True_value(dataset, metric, k):
    
    true_value = list()
    
    for model in models:

        true_value.append(get_True_metric(model, dataset, k, metric))

    true_value = np.array(true_value)
    
    return true_value

def get_True_max(dataset, metric, k):
    
    true_value = get_True_value(dataset, metric, k)
    ind = np.argmax(true_value)
    
    return ind



def get_estimate(model, dataset, metric, k, estimator, n = 100, repeats = 100):
    
    temp_list = list()
    #path = '../estimator_V0/save_PR/' +'sample%d/'%n +dataset + '/'+estimator+'/' + model + '/'
    
    path = '../save_PR/' +'fix_sample_%d/'%n +dataset + '/'+estimator+'/' + model + '/'
    
    for re in range(repeats):

        R = np.load(path + '%d.npz'%re)['R']
        metric_estimator = eval(metric + '_K(R, k)')
        
        temp_list.append(metric_estimator[-1])

    temp_array = np.array(temp_list)
    
    return temp_array

def get_estimate_adaptive(model, dataset, metric, k, repeats = 100):
    
    temp_list = list()

    
    path = '../save_PR/' +'adaptive_sample/' + dataset + '/MLE_adaptive/' + model + '/'
    
    for re in range(repeats):

        R = np.load(path + '%d.npz'%re)['R']
        metric_estimator = eval(metric + '_K(R, k)')
        
        temp_list.append(metric_estimator[-1])

    temp_array = np.array(temp_list)
    
    return temp_array

# MR determine the metric
def get_max(dataset, estimator, metric, k, sample_size):
    
    temp_list = list()
    
    for model in models:
        
        temp_list.append(get_estimate(model, dataset, metric, k, estimator, n = sample_size))
        
    temp_list = np.array(temp_list)
    
    temp_list = temp_list.transpose()
    
    return np.argmax(temp_list, -1)

def get_max_adaptive(dataset, metric, k):
    
    temp_list = list()
    
    for model in models:
        
        temp_list.append(get_estimate_adaptive(model, dataset, metric, k))
        
    temp_list = np.array(temp_list)
    
    temp_list = temp_list.transpose()
    
    return np.argmax(temp_list, -1)

def num_winner(dataset, estimator, metric, k, sample_size):
    
    
    true_ind = get_True_max(dataset, metric, k)
    
    es_ind = get_max(dataset, estimator, metric, k, sample_size)
    
    z = (es_ind == true_ind) * 1
    
    z = z.sum()

    return z

def num_winner_adaptive(dataset, metric, k):
    
    
    true_ind = get_True_max(dataset, metric, k)
    
    es_ind = get_max_adaptive(dataset, metric, k)
    
    z = (es_ind == true_ind) * 1
    
    z = z.sum()

    return z




nums = {'pinterest-20':9916, 'yelp':25815,  'ml-20m':20720}

datasets = ['pinterest-20','yelp','ml-20m']

models = ['EASE','MultiVAE','NeuMF','itemKNN','ALS']

metrics = ['RECALL', 'NDCG', 'AP']

estimators = ['MES','MLE','BV_MES','BV_MLE', 'MN_MES','MN_MLE']




dataset = 'yelp'

sample_size = 500 # 500

repeats = 100





ks = [10, 20]




f = open('results/' + dataset +'_winner_user.txt', 'w')

for k in ks:
    print(k)
    for metric in tqdm(metrics):

        for estimator in estimators:

            nn = num_winner(dataset, estimator, metric, k, sample_size)

            f.write(str(nn)+'\t')
            
        nn = num_winner_adaptive(dataset, metric, k)
        f.write(str(nn)+'\t')

        f.write('\n')
f.close()






