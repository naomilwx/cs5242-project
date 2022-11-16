import torch
import torch.nn as nn

from model.attention import AttentionBlock, AttentionConv
from model.conv_block import ConvBlock
from model.residual_cnn import Block

class CNNWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(CNNWithAttention, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 16, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2), #112x112

            nn.Conv2d(16, 32, (3,3), 1, 1),  
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
        )
        self.features2 = nn.Sequential(
            nn.MaxPool2d(2, 2),#56x56
            nn.Conv2d(32, 64, (3,3), 1, 1), 
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )
        self.features3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, (3,3), 1, 1), #28x28
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )
     
        self.attn1 = AttentionBlock(32, 128, 32, 4, normalize_attn=True)
        self.attn2 = AttentionBlock(64, 128, 64, 2, normalize_attn=True)

        self.pool = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(128+64+32, num_classes)

    
    def forward(self, x):
        feat1 = self.features1(x)
        feat2 = self.features2(feat1)
        feat3 = self.features3(feat2)
        out = self.pool(feat3).flatten(1)
        
        _, ct1 = self.attn1(feat1, feat3)

        _, ct2 = self.attn2(feat2, feat3)
        out = self.classifier(torch.cat((out, ct1, ct2), dim=1))

        return out

class ResidualCNNWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(ResidualCNNWithAttention, self).__init__()
        self.features1 = nn.Sequential(
            ConvBlock(3, 16),
            Block(16, 16),
            nn.MaxPool2d(2, 2),#112x112

            Block(16, 32),
            Block(32, 32),
            nn.MaxPool2d(2, 2), #56x56

            Block(32, 64),
            Block(64, 64)
        )
        self.features2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Block(64, 128),
            Block(128, 128),
        )
            
        self.attn = AttentionBlock(64, 128, 64, 2, normalize_attn=True)


        self.pool = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(128+64, num_classes)

    def forward(self, x):
        feat1 = self.features1(x)
        feat2 = self.features2(feat1)
        out = self.pool(feat2).flatten(1)

        _, ct = self.attn(feat1, feat2)
        out = self.classifier(torch.cat((out, ct), dim=1))

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
            nn.MaxPool2d(2, 2), #112X112

            nn.Conv2d(16, 32, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2), #56X56

            nn.Conv2d(32, 64, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2), # 28X28

            nn.Conv2d(64, 128, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            AttentionConv(128, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2), # 28X28,
        )

        self.aap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.aap(out)
        out = self.classifier(self.flatten(out))

        return out