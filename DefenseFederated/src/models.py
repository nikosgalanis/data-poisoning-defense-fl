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

### CLASSIC, AVAILABLE IN ALMOST ALL SOURCES
    
    # def __init__(self):
    #     super(CNNCifar, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 16 * 5 * 5)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return F.log_softmax(x, dim=1)

### CLASSIC, BUT ENHANCHED BY GPT (works well for lr=0.1, epochs = 50, with many clients (50%) and many epochs per client)
    # def __init__(self):
    #     super(CNNCifar, self).__init__()

    #     # Add another convolutional layer before pooling layers
    #     self.conv1 = nn.Conv2d(3, 16, 3)
    #     self.bn1 = nn.BatchNorm2d(16)
    #     self.pool = nn.MaxPool2d(2, 2)

    #     self.conv2 = nn.Conv2d(16, 32, 5)
    #     self.bn2 = nn.BatchNorm2d(32)
    #     self.pool = nn.MaxPool2d(2, 2)

    #     # Fully connected layers
    #     self.fc1 = nn.Linear(32 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)

    # def forward(self, x):
    #     # Convolutional layers
    #     x = self.bn1(F.relu(self.conv1(x)))
    #     x = self.pool(x)

    #     x = self.bn2(F.relu(self.conv2(x)))
    #     x = self.pool(x)

    #     # Flatten the input
    #     x = x.view(-1, 32 * 5 * 5)

    #     # Fully connected layers
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)

    #     # Log softmax for classification
    #     return F.log_softmax(x, dim=1)

### CLASSIC, BUT ENHANCHED BY GPT (works well for lr=0.001, epochs = 50
    # def __init__(self):
    #     super(CNNCifar, self).__init__()

    #     # Convolutional layers
    #     self.conv1 = nn.Conv2d(3, 128, 3, padding=1)  # Added padding
    #     self.bn1 = nn.BatchNorm2d(128)
    #     self.conv2 = nn.Conv2d(128, 256, 3, padding=1)  # Adjusted kernel size and added padding
    #     self.bn2 = nn.BatchNorm2d(256)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.dropout1 = nn.Dropout(0.1)  # Added dropout

    #     # Fully connected layers
    #     self.fc1 = nn.Linear(256 * 8 * 8, 1024)  # Adjusted dimensions
    #     self.fc2 = nn.Linear(1024, 512)
    #     self.dropout2 = nn.Dropout(0.1)  # Added dropout
    #     self.fc3 = nn.Linear(512, 10)

    # def forward(self, x):
    #     # Convolutional layers
    #     x = self.pool(F.relu(self.bn1(self.conv1(x))))
    #     x = self.pool(F.relu(self.bn2(self.conv2(x))))
    #     x = self.dropout1(x)

    #     # Flatten the input
    #     x = x.view(-1, 256 * 8 * 8)  # Adjusted flattening

    #     # Fully connected layers
    #     x = F.relu(self.fc1(x))
    #     x = self.dropout2(x)
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)

    #     return F.log_softmax(x, dim=1)
    
### MOST ENHANCED BY GPT

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