import torch.nn as nn

class BaselineCNN1(nn.Module):

    def __init__(self, num_classes):
        super(BaselineCNN1, self).__init__()

        self.c3_16 = nn.Conv2d(3, 16, (3,3), 1, 1)
        self.c16_16 = nn.Conv2d(16, 16, (3,3), 1, 1)
        self.c16_32 = nn.Conv2d(16, 32, (3,3), 1, 1)
        self.c32_32 = nn.Conv2d(32, 32, (3,3), 1, 1)
        self.c32_64 = nn.Conv2d(32, 64, (3,3), 1, 1)
        self.c64_64 = nn.Conv2d(64, 64, (3,3), 1, 1)
        self.c64_128 = nn.Conv2d(64, 128, (3,3), 1, 1)
        self.c128_128 = nn.Conv2d(128, 128, (3,3), 1, 1)
        self.relu = nn.ReLU()

        self.mp = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(128)
        self.aap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc128_out = nn.Linear(128, num_classes)

    
    def forward(self, x):
        out = self.relu(self.c3_16(x))
        out = self.relu(self.c16_16(out))
        out = self.mp(out)
        out = self.relu(self.c16_32(out))
        out = self.relu(self.c32_32(out))
        out = self.mp(out)
        out = self.relu(self.c32_64(out))
        out = self.relu(self.c64_64(out))
        out = self.mp(out)
        out = self.relu(self.c64_128(out))
        out = self.relu(self.c128_128(out))
        out = self.mp(out)
        out = self.aap(out)
        out = self.flatten(out)
        out = self.fc128_out(out)
        return out