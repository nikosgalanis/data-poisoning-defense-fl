import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import numpy as np

class DatasetSplit(Dataset):
    def __init__(self, dataset, clients):
        self.dataset = dataset
        self.clients = clients

    def __len__(self):
        return len(self.clients)

    def __getitem__(self, item):
        image, label = self.dataset[self.clients[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, dataset, clients):
        self.train_loader, self.validation_loader, self.test_loader = self.train_val_test(
            dataset, list(clients))
        
        self.device = 'cuda:0'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, clients):
        # split indexes for train, validation, and test (75, 10, 15)
        clients_train = clients[0 : int(0.75 * len(clients))]
        clients_val = clients[int(0.75 * len(clients)) : int(0.85 * len(clients))]
        clients_test = clients[int(0.85 * len(clients)):]
        
        train_loader = DataLoader(DatasetSplit(dataset, clients_train),
                                 batch_size=10, shuffle = True)
        
        validation_loader = DataLoader(DatasetSplit(dataset, clients_val),
                                 batch_size=int(len(clients_val) / 10), shuffle = False)
        
        test_loader = DataLoader(DatasetSplit(dataset, clients_test),
                                batch_size=int(len(clients_test) / 10), shuffle = False)
        
        return train_loader, validation_loader, test_loader

    def update_weights(self, model, local_epochs, learning_rate, fake=False):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        for _ in range(local_epochs):
            batch_loss = []
            
            for _, (images, labels) in enumerate(self.train_loader):
                
                # transfer the tensors to the appropriate device
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()

                # in the case of a fake update, we want the loss but
                # we don't want the optimizer to proceed
                if fake == False:
                    optimizer.step()

                batch_loss.append(loss.item())
            
            # compute the epoch loss and append it
            e_loss = sum(batch_loss) / len(self.train_loader)
            epoch_loss.append(e_loss)

        # return the state of the model, together with the total loss
        return model.state_dict(), sum(epoch_loss) / local_epochs


    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        all_labels = []
        all_pred_labels = []

        with torch.no_grad():
            for _, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

                # Append the batch results
                all_labels.extend(labels.cpu().numpy())
                all_pred_labels.extend(pred_labels.cpu().numpy())

        accuracy = correct / total
        
        # Compute the confusion matrix
        confusion_mat = confusion_matrix(all_labels, all_pred_labels)
        class_id = 3
        if np.sum(confusion_mat[class_id, :]) > 0:
            source_class_recall = confusion_mat[class_id, class_id] / np.sum(confusion_mat[class_id, :])
        else:
            source_class_recall = 0
            
        return accuracy, loss, source_class_recall  


def test_inference(model, test_dataset, target_mal):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda:0'

    criterion = nn.NLLLoss().to(device)
    test_loader = DataLoader(test_dataset, batch_size=64,
                            shuffle=False)

    all_labels = []
    all_pred_labels = []
    
    with torch.no_grad():
        for _, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            # Append the batch results
            all_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(pred_labels.cpu().numpy())
            
    accuracy = correct / total
    
    # Compute the confusion matrix
    confusion_mat = confusion_matrix(all_labels, all_pred_labels)
    class_id = target_mal
    if np.sum(confusion_mat[class_id, :]) > 0:
        source_class_recall = confusion_mat[class_id, class_id] / (np.sum(confusion_mat[class_id, :]) + 0.001)
    else:
        source_class_recall = 0
        
    return accuracy, loss / len(test_loader), source_class_recall

