import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import argparse



datasets = ['pinterest-20','yelp','ml-20m']


path = './processed_data/'


sample_size = 99 # negative sample size is 99, 499
repeatings = 10

for dataset in datasets:
    
    save_path = './fix_sample_%d/'%(sample_size + 1) + dataset + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('processing %s'%dataset)

    

    #np.random.seed(9876532)

    train_path = path + dataset + '_train.csv'
    test_path = path + dataset + '_test.csv'
    valid_path = path + dataset + '_valid.csv'
    
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)
    
    

    num_user = max(train_df['user'].values.max(), test_df['user'].values.max()) 
    num_item = max(train_df['item'].values.max(), test_df['item'].values.max())
    
    valid_user = valid_df['user'].values.max()
    valid_item = valid_df['item'].values.max()
    
    num_user = max(num_user, valid_user) + 1
    num_item = max(num_item, valid_item) + 1
    

    #train_u_dict = train_df.groupby('user')['item'].apply(list).to_dict()
    #valid_u_dict = valid_df.groupby('user')['item'].apply(list).to_dict()
    test_u_dict = test_df.groupby('user')['item'].apply(list).to_dict()
    
    
    new_test_dict = {}
   
    for n in tqdm(range(repeatings)):
        np.random.seed(sample_size * 3 + n)
        
        for u, items in (test_u_dict.items()):
            i = items[0]
            value = [i]

            while(True):
                negative_items = np.random.choice(num_item, sample_size + 100, replace = True)
                negative_items = negative_items[negative_items != i]
                if len(negative_items) >= sample_size:
                    negative_items = negative_items[:sample_size].tolist()
                    break
            value.extend(negative_items)
    #         for t in range(sample_size):# sample negative items
    #             neg_i = np.random.randint(0, num_item)
    #             while neg_i == i:
    #                 neg_i = np.random.randint(0, num_item)
    #             value.append(neg_i)

            new_test_dict[u] = value


        np.save(save_path + '%d.npy'%(n), new_test_dict)