import torch.nn as nn
import torch.nn.functional as F
import torch

#https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
# The network should inherit from the nn.module
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # Convolutional layers (5)
        # parameters(input channels, output channels, kernel size, stride)
        self.conv1 = nn.Conv2d(1, 64, 11)
        self.conv2 = nn.Conv2d(64, 192, 5)
        self.conv3 = nn.Conv2d(192, 384, 3)
        self.conv4 = nn.Conv2d(384, 256, 3)
        self.conv5 = nn.Conv2d(256, 256, 3)

        # Max Pooling
        # paramters(kernel size)
        self.maxP1 = nn.MaxPool2d(3)
        self.maxP2 = nn.MaxPool2d(3)
        self.maxP3 = nn.MaxPool2d(3)

        # Fully Connected layers (3)
        # parameters(input size, output size)
        self.fc1 = nn.Linear(6400, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x): # Linking all the layers together
        x = self.conv1(x)
        x = F.relu(x) # activation function
        x = self.maxP1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxP2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.maxP3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.softmax(x, dim=1)  # Softmax used for probability training
        return output