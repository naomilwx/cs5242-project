import torch.nn as nn
import torch.nn.functional as F
from utils import device_utils

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))
        else:
          self.skip = None

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            )
        
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.block(x)

        if self.skip is not None:
          out = self.skip(x)
        else:
          out = x

        out += x

        out = F.relu(out)
        out = self.bn(out)
        return out

class BaselineCNN2(nn.Module):

    def __init__(self, num_classes):
        super(BaselineCNN2, self).__init__()

        self.features = nn.Sequential(
            Block(3, 64, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(64, 128, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(128, 256, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        
        self.aap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(256, num_classes)

    
    def forward(self, x):

        x = self.features(x)
            
        out = self.aap(out)
        out = self.flatten(out)
        out = self.classifier(out)
        return out