import numpy as np


sample_size_list = [100, 200, 400, 800, 1600]
init_size = 100 # 100 negative items

def get_adaptive_rank(u_dict)
"""
u_dict: test set, key is the user id, value is the test item id (assume only 1 test item )
"""
    sample_rank, sample_size = [], []
    for user, item in u_dict.items():

        neg_items = np.random.choice(num_item, size = init_size, replace = True)

        current_rank = f(user, item, neg_items) # give the rank of test item among neg_items.
        current_size = init_size
        if current_rank == 0:
            for sample_size in sample_size_list:

                new_neg_items = np.random.choice(num_item, size = sample_size, replace = True)
                current_rank += f(user, item, new_neg_items)
                current_size += sample_size
                if current_rank != 0:
                    break
        sample_rank.append(current_rank)
        sample_size.append(current_size)

    sample_rank = np.array(sample_rank).astype('int')
    sample_size = np.array(sample_size).astype('int')

    np.savez("adaptive.npz", size = sample_size, rank = sample_rank)
