#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import split_dataset

def get_dataset(args, mal_usr_percentage, target_hon, target_mal):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train = True, download = True, transform = apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train = False, download = True, transform = apply_transform)

        # sample training data amongst users
        train_dataset, user_groups, attackers = split_dataset(train_dataset, args.num_users, mal_usr_percentage, target_hon, target_mal)
 
    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train = True, download = True, transform = apply_transform)

        test_dataset = datasets.MNIST(data_dir, train = False, download = True, transform = apply_transform)

        # sample training data amongst users
        train_dataset, user_groups, attackers = split_dataset(train_dataset, args.num_users, mal_usr_percentage, target_hon, target_mal)

    return train_dataset, test_dataset, user_groups, attackers


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


# TODO: delete
def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def find_largest_diff_index(nums):
    # Initializing variables for max difference and its index
    max_diff = float('-inf')
    idx = -1

    # Iterating through the list
    for i in range(1, len(nums)):
        diff = nums[i] - nums[i - 1]
        if diff > max_diff:
            max_diff = diff
            idx = i

    return idx

