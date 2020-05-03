import torch.nn as nn
import torch.nn.functional as F
import torch

# The network should inherit from the nn.Module
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        
        # Convolutional layers (16) 3x3
        # 1: input channels 32: output channels, 3: kernel size, 1: stride
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1)

        # Max Pooling
        # parameters: (kernel size)
        self.maxP1 = nn.MaxPool2d(3)
        self.maxP2 = nn.MaxPool2d(3)

        # Fully connected layers (3)
        # parameters: (input size, output size)
        self.fc1 = nn.Linear(856, 594)
        self.fc2 = nn.Linear(594, 128)
        self.fc3 = nn.Linear(128, 10)

    # First Set of convolutions
    def conv2max1(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxP1(x)
        return x

    # Second Set of convolutions
    def conv3max1(self, x):
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.maxP2(x)
        return x

    def forward(self, x):
        x = conv2max1(self, x)
        x = conv2max1(self, x)
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.maxP1(x)
        x = conv3max1(self, x)
        x = conv3max1(self, x)
        x = conv3max1(self, x)
        # x = self.conv3(x)
        # x = F.relu(x)
        # x = self.conv4(x)
        # x = F.relu(x)
        # x = self.conv5(x)
        # x = F.relu(x)
        # x = self.maxP2(x)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        # Making Full Connections
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=1) # Can test logSoftmax vs. Softmax
        # output = F.softmax(x, dim=1)
        return output
