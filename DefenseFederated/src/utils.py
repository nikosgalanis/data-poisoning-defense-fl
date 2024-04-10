import copy
import torch
from torchvision import datasets, transforms
from sampling import split_dataset
import numpy as np
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

def get_dataset(mal_usr_percentage, target_hon, target_mal, n_total_clients, n_train_clients, dataset):
    if dataset == 'mnist':
        data_dir = '../data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        # split and define the train and test datasets
        train_dataset = datasets.MNIST(data_dir, train = True, download = True, transform = apply_transform)

        test_dataset = datasets.MNIST(data_dir, train = False, download = True, transform = apply_transform)

        train_dataset, user_groups, attackers = split_dataset(train_dataset, n_total_clients, n_train_clients, mal_usr_percentage, target_hon, target_mal)
    
    elif dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), 
            (0.2023, 0.1994, 0.2010))])

        # split and define the train and test datasets
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

        train_dataset, user_groups, attackers = split_dataset(train_dataset, n_total_clients, n_train_clients, mal_usr_percentage, target_hon, target_mal)


    return train_dataset, test_dataset, user_groups, attackers

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

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

def calculate_accuracy(actual, predicted, total_clients):
    TP = len(set(actual) & set(predicted))
    FP = len(set(predicted) - set(actual))
    FN = len(set(actual) - set(predicted))
    TN = len(set(total_clients) - set(actual) - set(predicted))

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    return accuracy

def calculate_f1_score(actual, predicted):
    TP = len(set(actual) & set(predicted))
    FP = len(set(predicted) - set(actual))
    FN = len(set(actual) - set(predicted))

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    if precision + recall == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)

def eliminate_fixed_percentage(info,n_train_clients, percentage):
    
    (local_losses, local_weights, selected_users) = info
    
    sorted_indices = sorted(range(len(selected_users)), key=lambda k: local_losses[k])

    local_losses = [local_losses[i] for i in sorted_indices]
    local_weights = [local_weights[i] for i in sorted_indices]
    selected_users = [selected_users[i] for i in sorted_indices]

    idx = int((1 - percentage) * n_train_clients)
    
    attackers = selected_users[idx:]
    selected_users = selected_users[:idx]

    
    return selected_users, attackers

def eliminate_largest_diff(info, n_train_clients):
    
    (local_losses, local_weights, selected_users) = info
    
    sorted_indices = sorted(range(len(selected_users)), key=lambda k: local_losses[k])

    local_losses = [local_losses[i] for i in sorted_indices]
    local_weights = [local_weights[i] for i in sorted_indices]
    selected_users = [selected_users[i] for i in sorted_indices]

    idx = find_largest_diff_index(local_losses[int(n_train_clients / 2):])
    idx += int((n_train_clients / 2) - 1)

    attackers = selected_users[idx:]
    
    selected_users = selected_users[:idx]
    
    return selected_users, attackers

def eliminate_with_z_score(info, threshold):
    (local_losses, local_weights, selected_users) = info
    
    sorted_indices = sorted(range(len(selected_users)), key=lambda k: local_losses[k])

    local_losses = [local_losses[i] for i in sorted_indices]
    local_weights = [local_weights[i] for i in sorted_indices]
    selected_users = [selected_users[i] for i in sorted_indices]
    
    mean = np.mean(local_losses)
    std = np.std(local_losses)
    z_scores = [(loss - mean) / std for loss in local_losses]
    
    attackers = np.where(np.abs(z_scores) > threshold)[0]
    
    selected_users_new = [selected_users[i] for i in range(len(selected_users)) if i not in attackers]
    
    return selected_users_new, attackers

def eliminate_kmeans(info):
    (local_losses, local_weights, selected_users) = info
    data = np.array(local_losses).reshape(-1, 1)
    # Apply KMeans clustering with 2 clusters
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)
    # Get cluster assignments and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    # We assume the cluster with the higher centroid value corresponds to attackers.
    # This is because attackers might have higher loss values than honest clients.
    attacker_cluster = np.argmax(centroids)
    # Get the indices of the attacker cluster
    attackers = np.where(labels == attacker_cluster)[0]
    selected_users_new = [selected_users[i] for i in range(len(selected_users)) if i not in attackers]
    
    return selected_users_new, attackers

def apply_ldp(losses, epsilon, sensitivity=0.001):
    # Compute the scale of the Laplace noise
    b = sensitivity / epsilon

    # Generate Laplace noise for each data point
    noise = np.random.laplace(0, b, len(losses))

    # Add the noise to the original data
    noisy_data = losses + noise

    return noisy_data