import numpy as np
import random
def split_dataset(dataset, num_users, n_train_clients, mal_usr_percentage, target_honest, target_mal):
    num_items = int(len(dataset) / num_users)
    dict_users = {}
    
    num_attackers = int(mal_usr_percentage * n_train_clients)
    attackers = set(range(num_attackers))  # First 'num_attackers' users are attackers
    # attackers = set(random.sample(range(n_train_clients), num_attackers))

    new_dataset = {}
    
    for i in range(num_users):
        start_idx = i * num_items
        end_idx = (i + 1) * num_items
        dict_users[i] = set(range(start_idx, end_idx))
        
        if i in attackers:
            for idx in dict_users[i]:
                if dataset[idx][1] == target_honest:
                    new_dataset[idx] = (dataset[idx][0], target_mal)
                else:
                    new_dataset[idx] = dataset[idx]
        else:
            for idx in dict_users[i]:
                new_dataset[idx] = dataset[idx]
                                
    return new_dataset, dict_users, attackers