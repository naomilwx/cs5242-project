import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_dim=64):
        super(BaselineCNN, self).__init__()
        
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
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3,3), 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )        
        self.aap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.text_classifier = nn.Linear(embed_dim, num_classes)

        self.classifier = nn.Linear(num_classes+128, num_classes)

    def forward(self, x, text, text_offsets):
        imfeat = self.aap(self.features(x))
        imfeat = self.flatten(imfeat)
        
        tout = self.text_classifier(self.embedding(text, text_offsets))
        return self.classifier(torch.cat((tout, imfeat), 1))

class DeeperCNN(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_dim=64):
        super(DeeperCNN, self).__init__()
        
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
        self.aap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.text_classifier = nn.Linear(embed_dim, num_classes)

        self.classifier = nn.Linear(num_classes+128, num_classes)

    def forward(self, x, text, text_offsets):
        imfeat = self.aap(self.features(x))
        imfeat = self.flatten(imfeat)
        
        tout = self.text_classifier(self.embedding(text, text_offsets))
        return self.classifier(torch.cat((tout, imfeat), 1))
