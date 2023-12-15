import copy
import numpy as np
from tqdm import tqdm

import torch

from update import LocalUpdate, test_inference
from models import CNNMnist
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
    client_lr = 0.01
    client_epochs = 10

    torch.cuda.set_device(0)
    device = 'cuda'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups, attackers = get_dataset(mal_usr_percentage / 100, target_hon, target_mal, n_total_clients)

    # Training
    clients = []
    for i in range(n_train_epochs):
        clients.append(np.random.choice(range(n_total_clients), n_train_clients, replace=False))

    train_loss_total = []
    train_accuracy_total = []
    train_recall_total = []

    test_loss_total = []
    test_accuracy_total = []
    test_recall_total = []

    attacker_detection_ratio = []

    # if dataset == 'mnist':
    #     global_model = CNNMnist()

    # # Saving:
    # torch.save({
    #     'model': global_model,
    # }, 'mnist.pth')
    attack = [0, 1]
    recalls = []

    mal = [0, 10, 20, 30, 40]
    
    for att in attack:
        last_epoch_recall = []

        for mal_usr_percentage in mal:
            
            train_dataset, _, user_groups, attackers = get_dataset(mal_usr_percentage / 100, target_hon, target_mal, n_total_clients)

            checkpoint = torch.load('mnist.pth')
            global_model = checkpoint['model']
            global_model.load_state_dict(global_model.state_dict())

            attack_ratio = 0
            
            print("Running experiment with " + str(mal_usr_percentage) + "% malicious clients")

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
                        local_model = LocalUpdate(dataset=train_dataset, clients=user_groups[user])

                        w, loss = local_model.update_weights(
                            model=copy.deepcopy(global_model), local_epochs=client_epochs, learning_rate=client_lr, fake=True)
                        
                        local_weights_fake.append(copy.deepcopy(w))
                        local_losses_fake.append(copy.deepcopy(loss))

                    
                    local_losses_fake = apply_ldp(local_losses_fake, epsilon=1.0, sensitivity=0.0001)
                    
                    info = (local_losses_fake, local_weights_fake, selected_users)
                    
                    selected_users, attackers_found = eliminate_largest_diff(info, n_train_clients)
                                
                    count = sum(1 for item in attackers_found if item in attackers)
                    
                    if mal_usr_percentage > 0:
                        attack_ratio += (count / ((mal_usr_percentage / 100) * n_train_clients))
                                
                elif att == 0:
                    selected_users = clients[epoch]
                    
                local_weights, local_losses = [], []
                
                for user in selected_users:
                    local_model = LocalUpdate(dataset=train_dataset, clients=user_groups[user])

                    w, loss = local_model.update_weights(
                        model=copy.deepcopy(global_model), local_epochs=client_epochs, learning_rate=client_lr)

                    
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
                    local_model = LocalUpdate(dataset = train_dataset, clients = user_groups[client])
                    
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
                attack_ratio /= n_train_epochs
                if attack_ratio > 1:
                    attack_ratio = 1
                attacker_detection_ratio.append(attack_ratio)
                train_accuracy_total.append(train_accuracy)
                train_loss_total.append(train_loss)
                train_recall_total.append(train_recall)

                test_accuracy_total.append(test_accuracy)
                test_loss_total.append(test_loss)
                test_recall_total.append(test_recall)

        recalls.append(last_epoch_recall)
    
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')


    epochs = [x for x in range(0, n_train_epochs)]
    for cnt, model_rec in enumerate(test_recall_total):

        plt.plot(epochs, model_rec, label = str(mal[cnt]) + "% of mal users")


    plt.xlabel("Epochs")
    plt.ylabel("Source Class Recall")
    plt.legend()
    plt.savefig('save/rec_many_largest_diff.png')



    plt.figure()
    epochs = [x for x in range(0, n_train_epochs)]
    for cnt, model_loss in enumerate(test_loss_total):

        plt.plot(epochs, model_loss, label = str(mal[cnt]) + "% of mal users")


    plt.xlabel("Epochs")
    plt.ylabel("CrossEntropy Loss")
    plt.legend()
    plt.savefig('save/loss_many_largest_diff.png')


    plt.figure()
    epochs = [x for x in range(0, n_train_epochs)]
    for cnt, model_acc in enumerate(test_accuracy_total):

        plt.plot(epochs, model_acc, label = str(mal[cnt]) + "% of mal users")


    plt.xlabel("Epochs")
    plt.ylabel("Sparse Categorical Accuracy")
    plt.legend()
    plt.savefig('save/accs_many_largest_diff.png')

    
    
    plt.figure()
    
    plt.plot(mal[1:], attacker_detection_ratio[1:], 'o',  label = "Attackers detection ratio", linestyle = "-", linewidth = .4)
    plt.xlabel("Malicious users %")
    plt.ylabel("Percentage of attackers detected")
    plt.legend()
    plt.savefig('save/attackers_largest_diff.png')


    plt.figure()
    bar_width = 0.35

    r1 = np.arange(len(mal[1:]))
    r2 = [x + bar_width for x in r1]

    plt.bar(r1, recalls[0][1:], width=bar_width, edgecolor='grey', label='With Defense')
    plt.bar(r2, recalls[1][1:], width=bar_width, edgecolor='grey', label='Without Defense')

    plt.title('Effect of Defense Mechanism on Source Class Recall in Poisoning Attacks in FL')
    plt.xlabel('Malicious Users Percentage', fontweight='bold')
    plt.ylabel('Source Class Recall', fontweight='bold')

    plt.xticks([r + bar_width for r in range(len(recalls[0][1:]))], mal[1:])

    plt.legend()
    plt.savefig('save/comparison_largest_diff.png')