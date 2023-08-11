#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


def split_dataset(dataset, num_users, mal_usr_percentage, target_honest, target_mal):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    
    num_attackers = int(mal_usr_percentage * num_users)
    attackers = set(np.random.choice(range(num_users), num_attackers, replace=False))

    new_dataset = {}
    
    for i in range(num_users):
        # np.random.seed(1)

        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        
        all_idxs = list(set(all_idxs) - dict_users[i])

        if i in attackers:
            for idx in dict_users[i]:
                if dataset[idx][1] == target_honest:
                    new_dataset[idx] = (dataset[idx][0], target_mal)
                else:
                    new_dataset[idx] = dataset[idx]
                    
        else:
            for idx in dict_users[i]:
                new_dataset[idx] = dataset[idx]
                                
    return new_dataset, dict_users


# def split_dataset(dataset, num_users, mal_usr_percentage, target_honest, target_mal):
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset) / num_users)
#     dict_users = {}
    
#     num_attackers = int(mal_usr_percentage * num_users)
#     attackers = set(range(num_attackers))  # First 'num_attackers' users are attackers

#     new_dataset = {}
    
#     for i in range(num_users):
#         start_idx = i * num_items
#         end_idx = (i + 1) * num_items
#         dict_users[i] = set(range(start_idx, end_idx))
        
#         if i in attackers:
#             for idx in dict_users[i]:
#                 if dataset[idx][1] == target_honest:
#                     new_dataset[idx] = (dataset[idx][0], target_mal)
#                 else:
#                     new_dataset[idx] = dataset[idx]
#         else:
#             for idx in dict_users[i]:
#                 new_dataset[idx] = dataset[idx]
                                
#     return new_dataset, dict_users