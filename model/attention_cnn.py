import torch
import torch.nn as nn

from model.attention import SpatialAttention, AttentionConv2d

class CNNWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(CNNWithAttention, self).__init__()
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
            nn.Conv2d(128, 128, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            # nn.MaxPool2d(2, 2),
        )

        self.attn = SpatialAttention(128)

        self.aap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(128, num_classes)

    
    def forward(self, x):
        feat = self.features(x)
        out = self.aap(feat)

        _, ct = self.attn(feat, out)
        out = self.classifier(ct)

        return out

class CNNWithConvAttention(nn.Module):
    def __init__(self, num_classes):
        super(CNNWithConvAttention, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2), #84x84

            nn.Conv2d(16, 32, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2), #42x42

            nn.Conv2d(32, 64, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2), # 21x21

            # nn.Conv2d((64, 128, (3, 3), 1, 1),
            AttentionConv2d(64, 128, 3, 128, 8, 4),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            AttentionConv2d(128, 128, 3, 128, 8, 4, relative=True, shape=21),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
        )

        self.aap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.aap(out)
        out = self.classifier(self.flatten(out))

        return out