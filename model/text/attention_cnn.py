import torch
import torch.nn as nn

from model.attention import AttentionBlock

class CNNWithAttention(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_dim=64):
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

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.text_classifier = nn.Linear(embed_dim, num_classes)

        self.classifier = nn.Linear(num_classes+128+64+32, num_classes)

    
    def forward(self, x, text, text_offsets):
        feat1 = self.features1(x)
        feat2 = self.features2(feat1)
        feat3 = self.features3(feat2)
        out = self.pool(feat3).flatten(1)
        
        _, ct1 = self.attn1(feat1, feat3)
        _, ct2 = self.attn2(feat2, feat3)

        tout = self.text_classifier(self.embedding(text, text_offsets))

        return self.classifier(torch.cat((tout, out, ct1, ct2), dim=1))

