import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):

    # Implement ReNet to sweep over the input image
    # and extract features from it.
    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.patch_size = 2
        self.rnn_front = nn.LSTM(input_size*self.patch_size**2, self.patch_size**2, 1, batch_first=True)
        self.rnn_back = nn.LSTM(input_size*self.patch_size**2, self.patch_size**2, 1, batch_first=True)

    def forward(self, x):
        # Convert into patches
        num_patches = x.shape[-1] // self.patch_size
        x_front = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x_front = x_front.contiguous().view(x_front.size(0), num_patches**2, -1)
        x_back =  x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x_back = x_back.flip(3).contiguous().view(x_back.size(0), num_patches**2, -1)
        out_front, _ = self.rnn_front(x_front) 
        out_back, _ = self.rnn_back(x_back)
        out = torch.cat((out_front, out_back), dim=2)
        return out  
    
class CNNWithRNN(nn.Module):

    def __init__(self, num_classes):
        super(CNNWithRNN, self).__init__()
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


        self.rnn = RNN(128)

        self.aap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(25, num_classes)

    
    def forward(self, x):
        feat = self.features(x)
        out = self.rnn(feat)
        out = self.aap(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out