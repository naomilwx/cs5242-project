import torch
import torch.nn as nn

from model.attention import SpatialAttention

class CNNWithAttention(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_dim=64):
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
            nn.MaxPool2d(2, 2)
        )

        self.attn = SpatialAttention(128)

        self.aap = nn.AdaptiveAvgPool2d((1,1))
        
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.text_classifier = nn.Linear(embed_dim, num_classes)

        self.classifier = nn.Linear(num_classes+128, num_classes)

    def forward(self, x, text, text_offsets):
        feat = self.features(x)
        out = self.aap(feat)

        _, ct = self.attn(feat, out)

        tout = self.text_classifier(self.embedding(text, text_offsets))
        return self.classifier(torch.cat((tout, ct), 1))
