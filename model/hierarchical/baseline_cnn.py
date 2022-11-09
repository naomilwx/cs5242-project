import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_classes, num_groups):
        super(BaselineCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )
        
        self.aap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()

        self.gclassifier = nn.Linear(128, num_groups)
        self.classifier = nn.Linear(128+num_groups, num_classes)

    
    def forward(self, x):
        out = self.features(x)
        out = self.flatten(self.aap(out))

        gout = self.gclassifier(out)
        cout = self.classifier(torch.cat((out, gout), 1))
        return gout, cout
        