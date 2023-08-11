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

    dataset = 'mnist'
    # dataset = 'cifar'


    args = args_parser()
    exp_details(args)

    args.num_users = n_total_clients
    
    
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups

    train_dataset, test_dataset, user_groups = get_dataset(args, mal_usr_percentage / 100, target_hon, target_mal)

    # Training
    
    clients = []
    for i in range(n_train_epochs):
        clients.append(np.random.choice(range(n_total_clients), n_train_clients, replace=False))
        # clients.append(range(n_total_clients)[:n_train_clients])

    train_loss_total = []
    train_accuracy_total = []
    train_recall_total = []

    test_loss_total = []
    test_accuracy_total = []
    test_recall_total = []



    # if dataset == 'mnist':
    #     global_model = CNNMnist(args=args)
    # elif dataset == 'cifar':
    #     global_model = CNNCifar(args=args)

    # # Saving:
    # torch.save({
    #     'model': global_model,
    # }, 'mnist.pth')

    mal = [0, 10, 20, 40, 50]
    
    for mal_usr_percentage in mal:
    
        train_dataset, _, user_groups = get_dataset(args, mal_usr_percentage / 100, target_hon, target_mal)

        checkpoint = torch.load('mnist.pth')
        global_model = checkpoint['model']
        global_model.load_state_dict(global_model.state_dict())

        print(mal_usr_percentage)

        # Send the model to the device and then set it to train mode
        global_model.to(device)
        global_model.train()
    
        # hold the initial global weights
        global_weights = global_model.state_dict()
        


        train_loss = []
        train_accuracy = []
        train_recall = []

        
        test_loss = []
        test_accuracy = []
        test_recall = []

        for epoch in tqdm(range(n_train_epochs)):
            local_weights, local_losses = [], []
            
            global_model.train()
        
            selected_users = clients[epoch]


            for user in selected_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset, clients=user_groups[user])

                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model))
                
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            loss_avg = sum(local_losses) / len(selected_users)
            
            train_loss_total.append(loss_avg)
            
            # update global weights
            global_weights = average_weights(local_weights)
            # update global weights
            global_model.load_state_dict(global_weights)

            # Calculate avg training accuracy over all users at every epoch
            train_acc = 0
            train_rec = 0
            # evaluation mode of the model
            global_model.eval()


            for client in selected_users:
                local_model = LocalUpdate(args = args, dataset = train_dataset, clients = user_groups[client])
                
                acc, loss, rec = local_model.inference(model = global_model)
                train_acc += acc
                train_rec += rec
                
            train_acc /= len(selected_users)
            train_rec /= len(selected_users)
            
            train_accuracy.append(train_acc)
            train_recall.append(train_rec)

            # print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            # print(f'Training Loss : {np.mean(np.array(train_loss))}')
            # print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
            # print('Train Recall: {:.2f}% \n'.format(100 * train_recall[-1]))

            # Test inference after each epoch
            test_acc, test_ls, test_rec = test_inference(global_model, test_dataset)

            test_accuracy.append(test_acc)
            test_loss.append(test_ls)
            test_recall.append(test_rec)
        
        train_accuracy_total.append(train_accuracy)
        train_loss_total.append(train_loss)
        train_recall_total.append(train_recall)

        test_accuracy_total.append(test_accuracy)
        test_loss_total.append(test_loss)
        test_recall_total.append(test_recall)


        print(train_accuracy_total)
        print(test_accuracy_total)

        print(f' \n Results after {args.epochs} global rounds of training:')
        print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy_total[-1][-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_accuracy_total[-1][-1]))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')


    epochs = [x for x in range(0, n_train_epochs)]
    for cnt, model_rec in enumerate(test_recall_total):

        plt.plot(epochs, model_rec, label = str(mal[cnt]) + "% of mal users")


    plt.xlabel("Epochs")
    plt.ylabel("Source Class Recall")
    plt.legend()
    plt.savefig('save/rec_many.png')



    plt.figure()
    epochs = [x for x in range(0, n_train_epochs)]
    for cnt, model_loss in enumerate(test_loss_total):

        plt.plot(epochs, model_loss, label = str(mal[cnt]) + "% of mal users")


    plt.xlabel("Epochs")
    plt.ylabel("CrossEntropy Loss")
    plt.legend()
    plt.savefig('save/loss_many.png')


    plt.figure()
    epochs = [x for x in range(0, n_train_epochs)]
    for cnt, model_acc in enumerate(test_accuracy_total):

        plt.plot(epochs, model_acc, label = str(mal[cnt]) + "% of mal users")


    plt.xlabel("Epochs")
    plt.ylabel("Sparse Categorical Accuracy")
    plt.legend()
    plt.savefig('save/accs_many.png')

    
    # # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy_total[0])), train_accuracy_total[0], label='Train Accuracy')
    # plt.plot(range(len(test_accuracy_total[0])), test_accuracy_total[0], label='Test Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epochs')
    # plt.legend()    
    # plt.savefig('save/acc.png')