import torch.nn as nn
from model.conv_block import ConvBlock

class DeeperCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeeperCNN, self).__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.aap = nn.AdaptiveAvgPool2d((1,1))
        
        self.classifier = nn.Linear(256, num_classes)
        self.flatten = nn.Flatten()

    
    def forward(self, x):
        out = self.features(x)
        out = self.flatten(self.aap(out))
        out = self.classifier(out)
        return out


class DeeperCNN2(nn.Module):
    def __init__(self, num_classes):
        super(DeeperCNN2, self).__init__()
        
        self.features = nn.Sequential(
            ConvBlock2(3, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock2(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock2(32, 64, depth=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock2(64, 128, depth=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.aap = nn.AdaptiveAvgPool2d((1,1))
        
        self.classifier = nn.Linear(128, num_classes)
        self.flatten = nn.Flatten()

    
    def forward(self, x):
        out = self.features(x)
        out = self.flatten(self.aap(out))
        out = self.classifier(out)
        return out