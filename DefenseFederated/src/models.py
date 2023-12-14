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
    
class CNNCifar(nn.Module):
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

    def __init__(self):
        super(CNNCifar, self).__init__()

        # Add another convolutional layer before pooling layers
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Convolutional layers
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool(x)

        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(x)

        # Flatten the input
        x = x.view(-1, 32 * 5 * 5)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Log softmax for classification
        return F.log_softmax(x, dim=1)



    # def __init__(self):
    #     in_channels = 3
    #     dropout_rate = 0
    #     num_classes = 10
    #     super(CNNCifar, self).__init__()
    #     self.out_channels = 32
    #     self.stride = 1
    #     self.padding = 2
    #     self.layers = []
    #     in_dim = in_channels
    #     for _ in range(4):
    #         self.layers.append(
    #             nn.Conv2d(in_dim, self.out_channels, 3, self.stride, self.padding)
    #         )
    #         in_dim = self.out_channels
    #     self.layers = nn.ModuleList(self.layers)

    #     self.gn_relu = nn.Sequential(
    #         nn.GroupNorm(self.out_channels, self.out_channels, affine=True),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2),
    #     )
    #     num_features = (
    #         self.out_channels
    #         * (self.stride + self.padding)
    #         * (self.stride + self.padding)
    #     )
    #     self.dropout = nn.Dropout(dropout_rate)
    #     self.fc = nn.Linear(num_features, num_classes)

    # def forward(self, x):
    #     for conv in self.layers:
    #         x = self.gn_relu(conv(x))

    #     x = x.view(-1, self.num_flat_features(x))
    #     x = self.fc(self.dropout(x))
    #     return x

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features