import torchvision.datasets as datasets
import random
from tqdm import tqdm
import copy

def generate_malicious_dataset(dataset, mal_percentage, target_honest, target_mal):
	"""
	Perform label-flipping on a dataset, with a specific target
	- dataset: the dataset for the attack
	- mal_percentage: the percentage of the target class that we are going to misclassify
	- target_honest: the class that we want to change
	- target_mal: the new, malicious label of the target
	"""
    # if we are indeed performing the attack
	if mal_percentage > 0:
    	# create a copy so we dont harm the original
		flipped_dataset = copy.deepcopy(dataset)
		# parse the dataset
		for i in range(len(flipped_dataset)):
			# if we find the target label	
			if flipped_dataset.targets[i] == target_honest:
				# only flip mal% of the target labels
				prob = random.random()
				if prob < mal_percentage:
					flipped_dataset.targets[i] = target_mal
		# return the malicious dataset
		return flipped_dataset
	else:
    	# if we are not launching the attack, just return the original one
		return dataset