import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


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
    
class CNNCifar(nn.Module):

    def __init__(self):
        super(CNNCifar, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='leaky_relu')
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='leaky_relu')
        self.bn2 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.1)  # Fine-tuned dropout rate

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.dropout2 = nn.Dropout(0.4)  # Fine-tuned dropout rate
        self.fc2 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(0.1)  # Fine-tuned dropout rate
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01))
        x = self.dropout1(x)

        # Flatten the input
        x = x.view(-1, 256 * 8 * 8)

        # Fully connected layers
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.dropout3(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)