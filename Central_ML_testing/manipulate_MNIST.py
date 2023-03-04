import torchvision.datasets as datasets
import random
from tqdm import tqdm
import copy


"""
Perform label-flipping on a dataset, with a specific target
 - dataset: the dataset for the attack
 - mal_percentage: the percentage of the target class that we are going to misclassify
 - target_honest: the class that we want to change
 - target_mal: the new, malicious label of the target
"""
def generate_malicious_dataset(dataset, mal_percentage, target_honest, target_mal):
	flipped_dataset = copy.deepcopy(dataset)

	for i in tqdm(range(len(flipped_dataset))):
    		
		if flipped_dataset.targets[i] == target_honest:
			prob = random.random()
			if prob < mal_percentage:
				flipped_dataset.targets[i] = target_mal
	return flipped_dataset