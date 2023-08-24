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
from utils import *

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

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

    train_dataset, test_dataset, user_groups, attackers = get_dataset(args, mal_usr_percentage / 100, target_hon, target_mal)

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

    attacker_detection_ratio = []

    # if dataset == 'mnist':
    #     global_model = CNNMnist(args=args)
    # elif dataset == 'cifar':
    #     global_model = CNNCifar(args=args)

    # # Saving:
    # torch.save({
    #     'model': global_model,
    # }, 'mnist.pth')
    attack = [1]
    recalls = []
    recalls_fake = [0.8306922468393596, 0.8207912665433005, 0.7306923458491625, 0.6376231310662068]


    mal = [0, 10, 20, 30, 40]
    
    for att in attack:
        last_epoch_recall = []

        for mal_usr_percentage in mal:
            
            train_dataset, _, user_groups, attackers = get_dataset(args, mal_usr_percentage / 100, target_hon, target_mal)

            checkpoint = torch.load('mnist.pth')
            global_model = checkpoint['model']
            global_model.load_state_dict(global_model.state_dict())


            attack_ratio = 0
            
            print(mal_usr_percentage)

            # Send the model to the device and then set it to train mode
            global_model.to(device)
            global_model.train()
        
            # hold the initial global weights
            global_weights = global_model.state_dict()
            
            train_loss , train_accuracy, train_recall = [], [], []
            test_loss , test_accuracy, test_recall = [], [], []


            for epoch in tqdm(range(n_train_epochs)):
                if att == 1:
                    local_weights_fake, local_losses_fake = [], []
                    
                    global_model.train()
                
                    selected_users = clients[epoch]


                    for user in selected_users:
                        local_model = LocalUpdate(args=args, dataset=train_dataset, clients=user_groups[user])

                        w, loss = local_model.fake_update_weights(
                            model=copy.deepcopy(global_model))
                        
                        local_weights_fake.append(copy.deepcopy(w))
                        local_losses_fake.append(copy.deepcopy(loss))

                    
                    local_losses_fake = apply_ldp(local_losses_fake, epsilon=1.0, sensitivity=0.0001)
                    
                    info = (local_losses_fake, local_weights_fake, selected_users)
                    
                    selected_users, attackers_found = eliminate_fixed_percentage(info, n_train_clients, 0.7)
                                
                    count = sum(1 for item in attackers_found if item in attackers and item in selected_users)
                    if mal_usr_percentage > 0:
                        attack_ratio += (count / (mal_usr_percentage / 100 * n_train_clients))
                                

                elif att == 0:
                    selected_users = clients[epoch]
                    
                    
                local_weights, local_losses = [], []
                
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

                # Test inference after each epoch
                test_acc, test_ls, test_rec = test_inference(global_model, test_dataset)

                test_accuracy.append(test_acc)
                test_loss.append(test_ls)
                test_recall.append(test_rec)
                
            
            last_epoch_recall.append(test_recall[-1])
            
            if att == 1:
                attacker_detection_ratio.append(attack_ratio / n_train_epochs)
                train_accuracy_total.append(train_accuracy)
                train_loss_total.append(train_loss)
                train_recall_total.append(train_recall)

                test_accuracy_total.append(test_accuracy)
                test_loss_total.append(test_loss)
                test_recall_total.append(test_recall)

            # print(f' \n Results after {args.epochs} global rounds of training:')
            # print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy_total[-1][-1]))
            # print("|---- Test Accuracy: {:.2f}%".format(100 * test_accuracy_total[-1][-1]))

        recalls.append(last_epoch_recall)
    
    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')


    # epochs = [x for x in range(0, n_train_epochs)]
    # for cnt, model_rec in enumerate(test_recall_total):

    #     plt.plot(epochs, model_rec, label = str(mal[cnt]) + "% of mal users")


    # plt.xlabel("Epochs")
    # plt.ylabel("Source Class Recall")
    # plt.legend()
    # plt.savefig('save/rec_many_fix_percentage.png')



    plt.figure()
    epochs = [x for x in range(0, n_train_epochs)]
    for cnt, model_loss in enumerate(test_loss_total):

        plt.plot(epochs, model_loss, label = str(mal[cnt]) + "% of mal users")


    plt.xlabel("Epochs")
    plt.ylabel("CrossEntropy Loss")
    plt.legend()
    plt.savefig('save/loss_many_fix_percentage.png')


    plt.figure()
    epochs = [x for x in range(0, n_train_epochs)]
    for cnt, model_acc in enumerate(test_accuracy_total):

        plt.plot(epochs, model_acc, label = str(mal[cnt]) + "% of mal users")


    plt.xlabel("Epochs")
    plt.ylabel("Sparse Categorical Accuracy")
    plt.legend()
    plt.savefig('save/accs_many_fix_percentage.png')

    
    
    plt.figure()
    
    plt.plot(mal[1:], attacker_detection_ratio[1:], 'o',  label = "Attackers detection ratio", linestyle = "-", linewidth = .4)
    plt.xlabel("Malicious users %")
    plt.ylabel("Percentage of attackers detected")
    plt.legend()
    plt.savefig('save/attackers_fix_percentage.png')



    print(recalls[0][1:])

    # plt.figure()
    # # Bar width
    # bar_width = 0.35

    # # Positions of bars
    # r1 = np.arange(len(mal[1:]))
    # r2 = [x + bar_width for x in r1]

    # plt.bar(r1, recalls[1][1:], width=bar_width, edgecolor='grey', label='With Defense')
    # plt.bar(r2, recalls[0][1:], width=bar_width, edgecolor='grey', label='Without Defense')

    # # Title & Subtitle
    # plt.title('Effect of Defense Mechanism on Source Class Recall in Poisoning Attacks in FL')
    # plt.xlabel('Malicious Users Percentage', fontweight='bold')
    # plt.ylabel('Source Class Recall', fontweight='bold')

    # # x axis
    # plt.xticks([r + bar_width for r in range(len(recalls[0][1:]))], mal[1:])

    # # Create legend & Show graphic
    # plt.legend()
    # plt.savefig('save/comparison_fix_percentage.png')
    
    # # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy_total[0])), train_accuracy_total[0], label='Train Accuracy')
    # plt.plot(range(len(test_accuracy_total[0])), test_accuracy_total[0], label='Test Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epochs')
    # plt.legend()    
    # plt.savefig('save/acc.png')