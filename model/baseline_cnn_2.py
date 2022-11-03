import torch.nn as nn
from utils import device_utils

class BaselineCNN2(nn.Module):

    def __init__(self, num_classes):
        super(BaselineCNN2, self).__init__()

        self.features = [
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
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2)
        ]

        self.device = device_utils.get_device()

        for i, layer in enumerate(self.features):
            self.features[i] = layer.to(self.device)


        
        self.aap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(256, num_classes)

    
    def forward(self, x):

        # Adding skip connection
        activation_count = 0
        skipped = x
        out = x
        for layer in self.features:
            if(isinstance(layer, nn.ReLU)):
                activation_count += 1
                if(activation_count % 2 == 0):
                    out = out + skipped
            if(isinstance(layer, nn.MaxPool2d)):
                skipped = out
            out = layer(out)
            
            
        out = self.aap(out)
        out = self.flatten(out)
        out = self.classifier(out)
        return out