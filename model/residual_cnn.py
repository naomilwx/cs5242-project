import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2):
        super(Block, self).__init__()

        self.sample = None
        if in_channels != out_channels:
            self.sample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        
        blocks = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.BatchNorm2d(out_channels)
        ]
        for i in range(1, depth):
            blocks.extend([
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
                nn.ReLU(True),
                nn.BatchNorm2d(out_channels)
            ])
        self.block = nn.Sequential(*blocks)
        
    def forward(self, x):
        identity = x
        if self.sample is not None:
            identity = self.sample(x)
        
        out = self.block(x)

        return out + identity

class ResidualCNN(nn.Module):

    def __init__(self, num_classes):
        super(ResidualCNN, self).__init__()

        self.features = nn.Sequential(
            Block(3, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(64, 128),
            Block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
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

class DeeperResidualCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeeperResidualCNN, self).__init__()

        self.features = nn.Sequential(
            Block(3, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32, 64),
            Block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(64, 128),
            Block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(128, 256),
            Block(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
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