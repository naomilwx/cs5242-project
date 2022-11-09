import torch
import torch.nn as nn

from model.baseline_cnn_2 import Block

class ResidualCNN(nn.Module):

    def __init__(self, num_classes, vocab_size, embed_dim=64):
        super(ResidualCNN, self).__init__()

        self.features = nn.Sequential(
            Block(3, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(16,32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(32, 64, depth=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Block(64, 128, depth=3)
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