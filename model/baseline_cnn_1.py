import torch.nn as nn
from model.conv_block import ConvBlock

class BaselineCNN1(nn.Module):
    def __init__(self, num_classes):
        super(BaselineCNN1, self).__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.aap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(128, num_classes)

    
    def forward(self, x):
        out = self.features(x)
        out = self.aap(out)
        out = self.flatten(out)
        out = self.classifier(out)
        return out

class CNNToMLP(nn.Module):
    def __init__(self, num_classes):
        super(CNNToMLP, self).__init__()
        
        self.features = nn.Sequential(
            ConvBlock(3, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.aap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    
    def forward(self, x):
        out = self.features(x)
        out = self.aap(out)
        out = self.flatten(out)
        out = self.classifier(out)
        return out