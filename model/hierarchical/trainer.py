import torch
from torch.utils.data import random_split, DataLoader
import copy
from torchvision import transforms
from model.hierarchical.hiearchical_losses import HiearchicalLoss

from utils.device_utils import get_device

class Trainer:
    def __init__(self, model, optimizer, criterion, dataset, batch_size, split=[0.7, 0.1, 0.2], seed=42):
        self.model = model
        self.optimizer = optimizer
        self.loss_calc = HiearchicalLoss(criterion)
        self.batch_size = batch_size
        self.seed = seed
        self._initialise_dataloaders(dataset, split)
        self.device = get_device()
        self.best_model = None

    def set_device(self, device):
        self.device = device
    
    def _initialise_dataloaders(self, dataset, split):
        dataset_size = len(dataset)
        train_size = int(split[0] * dataset_size)
        valid_size = int(split[1] * dataset_size)
        test_size = dataset_size - train_size - valid_size
        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(self.seed))
        self.trainloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        self.validloader = DataLoader(valid_dataset, shuffle=True, batch_size=self.batch_size)
        self.testloader = DataLoader(test_dataset, shuffle=True, batch_size=self.batch_size)

    def run_train(self, num_epochs):
        model = self.model.to(self.device)
        model.train()

        best_acc = None
        best_epoch = None
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            total_loss = 0.0
            total = 0
            correct = 0
            gcorrect = 0
            for i, (inputs, groups, labels) in enumerate(self.trainloader, 0):
                inputs, groups, labels = inputs.to(self.device), groups.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                g_pred, l_pred = model(inputs)
                loss = self.loss_calc.total_loss(g_pred, l_pred, groups, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                total_loss += loss.item()

                predicted = torch.argmax(l_pred.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                gcorrect += (torch.argmax(g_pred.data, 1) == groups).sum().item()

                if (i + 1) % 100 == 0:
                    print('epoch {:3d} | {:5d} batches loss: {:.4f}'.format(epoch, i + 1, running_loss/100))
                    running_loss = 0.0

            train_acc = correct/total
            train_gacc = gcorrect/total
            print("[Epoch {:3d}]: Training loss: {:5f} | Accuracy: {:5f} | Group accuracy: {:5f}".format(epoch, total_loss/(i+1), train_acc, train_gacc))
            
            val_loss, val_acc, top_k, val_gacc = self.run_test(self.validloader, model=model)
            print("[Epoch {:3d}]: Validation loss: {:5f} | Accuracy: {:5f} | Within 3: {:5f} | Group accuracy: {:5f}".format(epoch, val_loss, val_acc, top_k, val_gacc))
            model.train()

            if best_acc is None or val_acc > best_acc:
                best_acc = val_acc
                self.best_model = copy.deepcopy(model).to(self.device)
                best_epoch = epoch
        print('Best epoch: ', best_epoch)

    def run_test(self, dataloader, k=3, with_stats=False, model=None):
        if model is None:
            model = self.best_model if self.best_model is not None else self.model
        model.eval()

        correct = 0
        gcorrect = 0
        within_k = 0
        total = 0
        incorrect = {}
        avg_loss = 0
        num_batch = 0
        with torch.no_grad():
            for inputs, groups, labels in dataloader:
                num_batch += 1
                inputs, groups, labels = inputs.to(self.device), groups.to(self.device), labels.to(self.device)
                g_pred, l_pred = model(inputs)
                _, predicted = torch.topk(l_pred.data, k, 1, largest=True, sorted=True)
                
                total += labels.size(0)
                correct += (predicted[:, 0] == labels).sum().item()
                within_k += torch.eq(labels.unsqueeze(1), predicted).sum().item()
                gcorrect += (torch.argmax(g_pred, 1) == groups).sum().item()

                avg_loss += self.loss_calc.total_loss(g_pred, l_pred, groups, labels).item()

                if with_stats:
                    idxes = (predicted[:, 0] != labels).nonzero().flatten()
                    for i in idxes:
                        key = (labels[i].item(), predicted[i][0].item())
                        if key in incorrect:
                            incorrect[key]+=1
                        else:
                            incorrect[key] = 1
        avg_loss /= num_batch
        acc = correct/total

        if with_stats:
            return avg_loss, acc, within_k/total, gcorrect/total, incorrect
        return avg_loss, acc, within_k/total, gcorrect/total