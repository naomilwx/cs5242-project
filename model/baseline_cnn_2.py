import torch.nn as nn
import torch.nn.functional as F
from utils import device_utils

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()

        self.skip = nn.Sequential(
          nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
          nn.BatchNorm2d(out_channels)
        )

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        skip = self.skip(x)
        
        out = self.block(x)
        out += skip

        out = F.relu(out)
        return out

class BaselineCNN2(nn.Module):

    def __init__(self, num_classes):
        super(BaselineCNN2, self).__init__()

        self.features = nn.Sequential(
            Block(3, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.aap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(256, num_classes)

    
    def forward(self, x):
        out = self.features(x)
        out = self.aap(out)
        out = self.flatten(out)
        out = self.classifier(out)
        return out