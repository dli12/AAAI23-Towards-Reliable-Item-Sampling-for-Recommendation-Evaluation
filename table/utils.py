import numpy as np
import scipy.stats as stats


import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

from tqdm import tqdm

def get_adaptive_estimate(model, dataset, estimator, n = 100, repeats = 100):
    
    temp_list = list()
    path = '../save_PR/' +'adaptive_sample/' +dataset + '/'+estimator+'/' + model + '/'
    for re in range(repeats):

        R = np.load(path + '%d.npz'%re)['R']
        temp_list.append(R)

    temp_array = np.array(temp_list)
    
    return temp_array


def get_fix_estimate(model, dataset, estimator, n = 100, repeats = 100):
    
    temp_list = list()
    path = '../save_PR/' +'fix_sample_%d/'%n +dataset + '/'+estimator+'/' + model + '/'
    for re in range(repeats):

        R = np.load(path + '%d.npz'%re)['R']
        temp_list.append(R)

    temp_array = np.array(temp_list)
    
    return temp_array

def load_True(model, dataset):
    
    
    path = "../results/fix_sample_100/" + dataset + '/' + model + '/'
    
    data = np.load(path + "0.npz")
    
    population = data['origin']
    sample = data['sample']
    
    return sample, population

def abs_relative(a, b):
    
    # replace where the b is zero
    
    idx = np.where(b == 0)[0]
    a[idx] = 1.
    b[idx] = 1.
    
    s = np.abs(a-b)/b
    
    return s.mean()

def rank_order(ru, size = 100):

    ranks = np.zeros(size)

    for r in ru:

        ranks[r] = ranks[r] + 1
        
    return ranks



def RECALL_K(PR, topk):
    V = PR[:topk].reshape(-1, 1)
    TL = np.tril(np.ones((topk, topk)))
    
    RK = TL@V
    
    return RK.reshape(-1)

def NDCG_K(PR, topk):
    
    V = PR[:topk].reshape(-1, 1)
    
    TL = np.tril(np.ones((topk, topk)))
    ndcg = np.reciprocal(np.log2(np.arange(1., topk + 1) + 1))
    TL = TL * ndcg
    
    RK = TL@V
    
    return RK.reshape(-1)

def AP_K(PR, topk):
    
    V = PR[:topk].reshape(-1, 1)
    
    TL = np.tril(np.ones((topk, topk)))
    ap = np.reciprocal((np.arange(1., topk + 1)))
  
    TL = TL * ap
    
    RK = TL@V
    
    return RK.reshape(-1)