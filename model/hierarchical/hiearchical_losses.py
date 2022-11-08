import torch

class HiearchicalLoss:
    def __init__(self, criterion, penalty=3, rel_weight=0.3):
        self.criterion = criterion
        self.rel_weight=0.3

    def prediction_loss(self, group_predictions, label_predictions, groups, labels):
        return self.criterion(group_predictions, groups) + self.criterion(label_predictions, labels)

    def hiearchical_loss(self, group_predictions, label_predictions, groups, labels):
        group_pred = torch.argmax(group_predictions, 1)
        label_pred = torch.argmax(label_predictions, 1)

        dl = (group_pred != groups).int()
        ll = (label_pred != labels).int()

        return torch.sum(torch.exp(dl)*torch.exp(dl*ll) - 1)

    def total_loss(self, group_predictions, label_predictions, groups, labels):
        return self.prediction_loss(group_predictions, label_predictions, groups, labels) + self.rel_weight * self.hiearchical_loss(group_predictions, label_predictions, groups, labels)