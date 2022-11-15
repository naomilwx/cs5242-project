import torch
import torch.nn as nn
import numpy as np

from model.trainer import Trainer

class HyperParameter:
    def __init__(self, lr, reg=0, transform=False, batch_size=32):
        self.lr = lr
        self.reg = reg
        self.random_transform=transform
        self.batch_size = batch_size

    def __str__ (self):
        return f"lr={self.lr}|reg={self.reg}|{self.random_transform}"

class ExperimentRunner:
    def __init__(self, model_class, hyperparameters):
        self.model_class = model_class
        self.results = {}
        self.seeds = [42, 141, 23, 131, 100]
        self.hyperparameters = hyperparameters

    def run(self, data, num_epochs):
        for seed in self.seeds:
            for param in self.hyperparameters:
                torch.manual_seed(seed)
                param_key = str(param)
                print(f"Running seed: {seed}, hyperparameters: {param_key}")
                if param_key not in self.results:
                    self.results[param_key] = {
                        'best': [],
                        'last': []
                    }
                model = self.model_class(len(data.categories))
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=param.lr, weight_decay=param.reg)

                mtrainer = Trainer(model, optimizer, criterion, data, param.batch_size, crop=1, random_transform=param.random_transform)
        
                mtrainer.run_train(num_epochs)

                _, test_acc, _ = mtrainer.run_test(mtrainer.testloader, 3)
                print(f"Accuracy: {test_acc}")

                _, test_acc_last, _ = mtrainer.run_test(mtrainer.testloader, model=mtrainer.model)

                self.results[param_key]['best'].append(test_acc)
                self.results[param_key]['last'].append(test_acc_last)
        return self.results
        
    def result_summary(self):
        results = {}
        std = {}
        for k, v in self.results.items():
            results[k] = {
                'best': np.mean(v['best']),
                'last': np.mean(v['last'])
            }
            std[k] = {
                'best': np.std(v['best']),
                'last': np.std(v['last'])
            }

        return results, std

                