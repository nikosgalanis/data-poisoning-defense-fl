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

### CLASSIC, BUT ENHANCHED BY GPT (works well for lr=0.1, epochs = 50, with many clients (50%) and many epochs per client)
    def __init__(self):
        super(CNNCifar, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 5)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 5 * 5, 1024)  # Adjusted dimensions
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 10)

    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Flatten the input
        x = x.view(-1, 256 * 5 * 5)  # Correctly flatten the tensor

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)  # Corrected final layer

        # Log softmax for classification
        return F.log_softmax(x, dim=1)
    
# ### 2ND TIME ENHANCHED BY GPT (works well for lr=0.1, epochs = 50, with many clients (50%) and many epochs per client)
## not working well, possibly too much dropout
#     def __init__(self):
#         super(CNNCifar, self).__init__()
#         # Improved convolutional layers with more depth and smaller kernel sizes
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)

#         # Use adaptive pooling to ensure the output size is fixed
#         self.pool = nn.AdaptiveMaxPool2d((5, 5))

#         # Dropout for regularization
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)

#         # Updated fully connected layers
#         self.fc1 = nn.Linear(128 * 5 * 5, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 10)

#     def forward(self, x):
#         # Convolutional layers
#         x = self.bn1(F.relu(self.conv1(x)))
#         x = self.bn2(F.relu(self.conv2(x)))
#         x = self.bn3(F.relu(self.conv3(x)))

#         # Adaptive pooling and dropout
#         x = self.pool(x)
#         x = self.dropout1(x)

#         # Flatten the input
#         x = x.view(-1, 128 * 5 * 5)

#         # Fully connected layers with dropout
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = F.relu(self.fc2(x))

#         # No activation needed before log_softmax
#         x = self.fc3(x)

#         return F.log_softmax(x, dim=1)


# ### by https://github.com/chart21/cifar10-shamir/blob/main/Fed_Avg_SMC.ipynb
#     def __init__(self):
#         super(CNNCifar, self).__init__()
#         # convolutional layer (sees 32x32x3 image tensor)
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         # convolutional layer (sees 16x16x16 tensor)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         # convolutional layer (sees 8x8x32 tensor)
#         self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
#         # max pooling layer
#         self.pool = nn.MaxPool2d(2, 2)
#         # linear layer (64 * 4 * 4 -> 500)
#         self.fc1 = nn.Linear(64 * 4 * 4, 500)
#         # linear layer (500 -> 10)
#         self.fc2 = nn.Linear(500, 10)
#         # # dropout layer (p=0.25)
#         # self.dropout = nn.Dropout(0.25)

#     def forward(self, x):
#         # add sequence of convolutional and max pooling layers
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         # flatten image input
#         x = x.view(-1, 64 * 4 * 4)
#         ## add dropout layer
#         # x = self.dropout(x)
#         # add 1st hidden layer, with relu activation function
#         x = F.relu(self.fc1(x))
#         ## add dropout layer
#         # x = self.dropout(x)
#         # add 2nd hidden layer, with relu activation function
#         x = self.fc2(x)
#         return x


### FACEBOOK - NOT WELL
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