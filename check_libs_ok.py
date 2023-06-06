import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


"""
Check if everything is ok with tf and tff
"""
print(tf.config.list_physical_devices('GPU'))


tff.federated_computation(lambda: 'Hello, World!')()

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
