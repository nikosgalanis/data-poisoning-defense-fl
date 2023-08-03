#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import numpy as np
from tqdm import tqdm

import torch

from options import args_parser
from update import LocalUpdate, test_inference
from models import CNNMnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

if __name__ == '__main__':

    n_train_epochs = 30
    n_train_clients = 50
    n_total_clients = 500


    mal_usr_percentage = 0
    target_hon = 3
    target_mal = 8

    # dataset = 'mnist'
    dataset = 'cifar'


    args = args_parser()
    exp_details(args)

    args.num_users = n_total_clients
    
    
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args, mal_usr_percentage, target_hon, target_mal)

    if dataset == 'mnist':
        global_model = CNNMnist(args=args)
    elif dataset == 'cifar':
        global_model = CNNCifar(args=args)


    # Send the model to the device and then set it to train mode
    global_model.to(device)
    global_model.train()
 
    # hold the initial global weights
    global_weights = global_model.state_dict()

    # Training
    train_loss = []
    train_accuracy = []
    train_recall = []
    
    test_loss = []
    test_accuracy = []
    test_recall = []
    
    for epoch in tqdm(range(30)):
        local_weights, local_losses = [], []
        
        global_model.train()
    
        selected_users = np.random.choice(range(n_total_clients), n_train_clients, replace=False)


        for user in selected_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, clients=user_groups[user])
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model))
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        loss_avg = sum(local_losses) / len(selected_users)
        
        train_loss.append(loss_avg)
        
        # update global weights
        global_weights = average_weights(local_weights)
        # update global weights
        global_model.load_state_dict(global_weights)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss, list_rec = [], [], []
        # evaluation mode of the model
        global_model.eval()
        
        for client in selected_users:
            local_model = LocalUpdate(args = args, dataset = train_dataset, clients = user_groups[client])
            
            acc, loss, rec = local_model.inference(model=global_model)
            
            list_acc.append(acc)
            list_loss.append(loss)
            list_rec.append(rec)
            
        train_acc = sum(list_acc) / len(selected_users)
        train_rec = sum(list_rec) / len(selected_users)
        
        train_accuracy.append(train_acc)
        train_recall.append(train_rec)

        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
        # print('Train Recall: {:.2f}% \n'.format(100 * train_recall[-1]))

        # Test inference after each epoch
        test_acc, test_ls, test_rec = test_inference(global_model, test_dataset)

        test_accuracy.append(test_acc)
        test_loss.append(test_ls)
        test_recall.append(test_rec)
        
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(test_loss)), test_loss, label='Test Loss')
    plt.plot(range(len(train_loss)), train_loss, label='Train loss')
    plt.ylabel('Crossentropy loss')
    plt.xlabel('Epochs')
    plt.legend()    
    
    plt.savefig('save/loss.png')

    
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, label='Train Accuracy')
    plt.plot(range(len(test_accuracy)), test_accuracy, label='Test Accuracy')
    plt.ylabel('SAccuracy')
    plt.xlabel('Epochs')
    plt.legend()    
    plt.savefig('save/acc.png')
    
    # Plot Average Recall vs Communication rounds
    plt.figure()
    plt.title('Average Recall vs Communication rounds')
    plt.plot(range(len(train_recall)), train_recall, label='Train Recall')
    plt.plot(range(len(test_recall)), test_recall, label='Test Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epochs')
    plt.legend()    
    plt.savefig('save/rec.png')
