import torch
from torch import nn
import torch.nn.functional as F


class CNNMnist(nn.Module):
    
    def __init__(self):
        super(CNNMnist, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3) # assuming input is grayscale image
        self.max_pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5408, 10) # output size needs to be calculated based on input size

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return F.log_softmax(x, dim=1)