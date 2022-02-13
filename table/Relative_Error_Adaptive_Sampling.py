


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


# # Load Data




nums = {'pinterest-20':9916, 'yelp':25815,  'ml-20m':20720}

datasets = ['pinterest-20','yelp','ml-20m']

models = ['EASE','MultiVAE','NeuMF','itemKNN','ALS']

estimators = ['MLE_adaptive']





sample_size = 100
repeats = 100





metric = 'NDCG'
topk = 50





f = open('results/' + metric +'_mean_std_adaptive.txt', 'w')

for e in estimators:
    f.write(e + '\t')
f.write('\n')

for dataset in (datasets):
    
    f.write(dataset + '\n')
    #print('dataset : %s'%dataset)
    for model in models:
        
        N = nums[dataset]
        
        ru, Ru =  load_True(model, dataset)

        M = len(Ru)# total number of users

        True_PR = rank_order(Ru, N)/M
        
        metric_true = eval(metric + '_K(True_PR, topk)')
        
        
        for estimator in tqdm(estimators):
   
            #PR_total = get_estimate(model, dataset, estimator)
            PR_total = get_adaptive_estimate(model, dataset, estimator, n = sample_size, repeats = repeats)
            
            error = list()
            
            for i in np.arange(repeats):
                
                PR = PR_total[i]
                metric_PR = eval(metric + '_K(PR, topk)')
                relative_error = abs_relative(metric_PR, metric_true)
                
                error.append(relative_error)
                
            errors = np.array(error)*100
            
            mean = errors.mean()

            mean = '{:.2f}'.format(round(mean, 2))
            #f.write(str(mean)+'\t')

            f.write(str(mean)+'zzzz')


            std = errors.std()

            std = '{:.2f}'.format(round(std, 2))

            f.write(str(std)+'\t')
      
        f.write('\n')
        
f.close()







