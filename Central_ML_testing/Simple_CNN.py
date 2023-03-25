import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from tqdm import tqdm


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

def train_epoch(optimizer, model, trainloader, criterion, device):
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward 
        outputs = model(inputs)
        # compute loss
        loss = criterion(outputs, labels)
        # pass gradients back
        loss.backward()
        # update parameters
        optimizer.step()
        preds = torch.argmax(outputs, axis=1)

        metrics_list = precision_recall_fscore_support(preds.cpu(), labels.cpu(), average='macro')

    return loss.item(), metrics_list[0], metrics_list[1], metrics_list[2]

def predict_test(model, testloader, device, criterion):

    # Test the network and print the classification report
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)


    return loss.item()

def predict_test_2(model, testloader, device, criterion):

    # Test the network and print the classification report
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.cpu()
            labels = labels.cpu()
            _, predicted = torch.max(outputs.data, 1)
            true_labels += labels.tolist()
            pred_labels += predicted.tolist()

    return true_labels, pred_labels