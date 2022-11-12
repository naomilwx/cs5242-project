import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2):
        super(ConvBlock, self).__init__()

        blocks = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.BatchNorm2d(out_channels)
        ]
        for i in range(1, depth):
            blocks.extend([
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
                nn.ReLU(True),
                nn.BatchNorm2d(out_channels),
            ])
        self.block = nn.Sequential(*blocks)
        
    def forward(self, x):       
        return self.block(x)

# We found that ConvBlock performs better - ReLU activation before batchnorm does better
class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2):
        super(ConvBlock2, self).__init__()

        blocks = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        ]
        for i in range(1, depth):
            blocks.extend([
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            ])
        self.block = nn.Sequential(*blocks)
        
    def forward(self, x):       
        return self.block(x)


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
        super(DeeperCNN, self).__init__()
        
        self.features = nn.Sequential(
            ConvBlock(3, 16, depth=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(16, 32, depth=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64, depth=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, depth=5),
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