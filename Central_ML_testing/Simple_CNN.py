import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self, hidden_neurons, drop_rate):
        super(SimpleCNN, self).__init__()
		# net structure: 2 convolutional layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		# a dropout layer
        self.conv2_drop = nn.Dropout2d(p=drop_rate)
		# a dense layer
        self.dense = nn.Linear(320, hidden_neurons)
        # another dropout layer
        self.dense_drop = nn.Dropout(p=drop_rate)
		# an output layer
        self.fc2 = nn.Linear(hidden_neurons, 10)

    def forward(self, x):
        # apply the 2 conv layers
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # apply the pooling and the droput layers
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # flatten the output of the convolutional procedure
        x = x.view(-1, 320)
        # fully connected layer
        x = self.dense(x)
        # the dropout layer
        x = self.dense_drop(x)
        # output layer
        x = self.fc2(x)
        
        # activation funciton
        return F.log_softmax(x)