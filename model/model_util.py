import pandas as pd
import os
import time
import numpy as np
import scipy.sparse as sps
import random

from torch.utils.data import Dataset, DataLoader
import torch


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

class BPRDataset(Dataset):

    def __init__(self, users, pos_items, neg_items):

        self.users = users
        self.pos_items = pos_items
        self.neg_items = neg_items

    def __len__(self):

        return len(self.users)

    def __getitem__(self, idx):

        user = self.users[idx]
        pos_item = self.pos_items[idx]
        neg_item = self.neg_items[idx]

        sample = {'user':user, 'pos_item':pos_item, 'neg_item':neg_item}

        return sample
    
class BCEDataset(Dataset):

    def __init__(self, users, items, labels):

        self.users = users
        self.items = items
        self.labels = labels

    def __len__(self):

        return len(self.users)

    def __getitem__(self, idx):

        user = self.users[idx]
        item = self.items[idx]
        label = self.labels[idx]

        sample = {'user':user, 'item':item, 'label':label}

        return sample
    
    

class DataReader(object):
    
    def __init__(self, path, dataset):
        
        train_path = path + dataset + '_train.csv'
        test_path = path + dataset + '_test.csv'
        
        self.train_df = pd.read_csv(train_path)
        #self.train_df.drop_duplicates(subset = ['user', 'item'], inplace = True)
        self.test_df = pd.read_csv(test_path)
        
        valid_path = path + dataset + '_valid.csv'
        valid_df = pd.read_csv(valid_path)
        valid_item = valid_df['item'].values.max()
        
        self.train_u_dict = self.train_df.groupby('user')['item'].apply(list).to_dict()
        self.test_u_dict = self.test_df.groupby('user')['item'].apply(list).to_dict()
        
        self.num_user = max(self.train_df['user'].values.max(), self.test_df['user'].values.max()) + 1
        self.num_item = max(self.train_df['item'].values.max(), self.test_df['item'].values.max()) + 1
        self.num_item = max(self.num_item -1, valid_item) + 1
        
            
            
        
        
    def get_user_item(self):
        
        return self.num_user, self.num_item
    
    def get_URM_train(self):
        
        row_id = self.train_df['user'].values
        col_id = self.train_df['item'].values
        data = np.ones_like(row_id)
        
        URM_train = sps.csr_matrix((data, (row_id, col_id)), shape = (self.num_user, self.num_item), dtype = 'float64')
        
        #R_ui = (URM_train.toarray() != 0).astype(int)
        
        return URM_train
    
    def bpr_getTrain(self, N, train_batch_size, g):

        train_u = []
        train_pos_i = []
        train_neg_i = []
        
        u_list = self.train_df['user'].values
        i_list = self.train_df['item'].values
        #u_dict = train_df.groupby('user')['item'].apply(list).to_dict()
        
        for index in range(len(u_list)):
            
            u = u_list[index]
            i = i_list[index]
            train_u.extend([u]*(N))
            train_pos_i.extend([i]*(N))
        
            PositiveSet = set(self.train_u_dict[u]) 

            for t in range(N):# sample negative items
                neg_i = np.random.randint(0, self.num_item)
                while neg_i in PositiveSet:
                    neg_i = np.random.randint(0, self.num_item)
                train_neg_i.append(neg_i)

        train_dataset = BPRDataset(train_u, train_pos_i, train_neg_i)
        
        train_dataloader = DataLoader(train_dataset,
                                      batch_size = train_batch_size, 
                                      shuffle = True,
                                      num_workers = 4,
                                      pin_memory = True,
                                      worker_init_fn=seed_worker,
                                      generator=g,
                                     )

        return train_dataloader
    
    def bce_getTrain(self, N, train_batch_size, g):
        '''
        N: num of negative samples
        '''

        train_u = []
        train_i = []
        train_l = []
        
        u_list = self.train_df['user'].values.tolist()
        i_list = self.train_df['item'].values.tolist()
      

        for index in range(len(u_list)):
            u = u_list[index]
            i = i_list[index]
            train_u.extend([u]*(N+1))
            train_i.append(i)
            train_l.append(1)
            train_l.extend([0]*N)
            PositiveSet = set(self.train_u_dict[u]) 

            for t in range(N):# sample negative items
                neg_i = np.random.randint(0, self.num_item)
                while neg_i in PositiveSet:
                    neg_i = np.random.randint(0, self.num_item)
                train_i.append(neg_i)

        train_dataset = BCEDataset(train_u, train_i, train_l)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size = train_batch_size, 
                                      shuffle = True,
                                      num_workers = 4,
                                      pin_memory = True,
                                      worker_init_fn=seed_worker,
                                      generator=g,
                                     )

        return train_dataloader
    
def sample_eval(model, u_dict, device, topk = 10, user_flag = True):

    RECALL = []
    
    for user, items in u_dict.items():
        
        if user_flag:
            user_tensor = torch.tensor(user).to(device)
        else:
            user_tensor = user
            
        item_tensor = torch.tensor(items).to(device)

        predictions = model.evaluate_user(user_tensor)[items]
        predictions = predictions.view(-1).to(device)
       
        _, indices = torch.topk(predictions, topk)
        
        top_items = torch.take(item_tensor, indices).cpu().numpy()

        p_item = items[0]

        if p_item in top_items:
            RECALL.append(1)
        else:
            RECALL.append(0)

    return np.mean(RECALL)


    
    
    
def save_model(model, dataset):
    
    path = "saved_model/" + dataset
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    # save best model
    name = model.__class__.__name__
    torch.save(model, os.path.join(path, name + '.pt'))
    
    
def save_Repeat(path, s, g, i):
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    np.savez(path + '%d.npz'%i, sample = s, origin = g)
    
def save_Adaptive(dataset, model_name, s, g, sz, new_dict, i):
    
    path = '../results/adaptive_sample/' + dataset +'/' + model_name + '/'
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    np.savez(path + '%d.npz'%i, sample = s, origin = g, size = sz)
    
    
    dict_path = '../data/adaptive_sample/' + dataset +'/' + model_name + '/'
    
    if not os.path.exists(dict_path):
        os.makedirs(dict_path)
    
    np.save(dict_path + '%d.npy'%(i), new_dict)
        
        
    