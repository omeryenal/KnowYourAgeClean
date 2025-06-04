import torch
import torch.nn as nn
import torch.nn.functional as F

class AgeRegressionCNN(nn.Module):
    def __init__(self):
        super(AgeRegressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv3(x)))  # [B, 64, 8, 8]
        x = torch.flatten(x, 1)               # Flatten
        x = self.dropout(x)                   # Dropout
        x = F.relu(self.fc1(x))               # Fully connected
        x = self.fc2(x)
        return x.view(-1)
