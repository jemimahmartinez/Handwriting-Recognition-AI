import torch.nn as nn
import torch.nn.functional as F
import torch

# The network should inherit from the nn.module
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # Convolutional layers (5)
        # parameters(input channels, output channels, kernel size, stride)
        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d()
        self.conv3 = nn.Conv2d()
        self.conv4 = nn.Conv2d()
        self.conv5 = nn.Conv2d()

        # Max Pooling
        # paramters(kernel size)
        self.maxP1 = nn.MaxPool2d()
        self.maxP2 = nn.MaxPool2d()
        self.maxP3 = nn.MaxPool2d()

        # Fully Connected layers (3)
        # parameters(input size, output size)
        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        self.fc3 = nn.Linear()

    def forward(self, x):
        x = self.conv1(x)
        x = F.ReLu(x) # activation function
        x = self.maxP1
        x = self.conv2(x)
        x = F.ReLu(x) # activation function
        x = self.maxP2(x)
        x = self.conv3(x)
        x = F.ReLu(x)
        x = self.conv4(x)
        x = F.ReLu(x)
        x = self.conv5(x)
        x = F.ReLu(x)
        x = self.maxP3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.ReLu(x)
        x = self.fc2(x)
        x = F.ReLu(x)
        x = self.fc3(x)
        output = F.Softmax(x, dim=1)
        return output



