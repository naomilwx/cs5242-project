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
