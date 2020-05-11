import torch.nn as nn
import torch.nn.functional as F
import torch
# The network should inherit from the nn.module
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Convolutional layers (2) 5x5
        # parameters: (input channels, output channels, kernel size, stride)
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        # Average pooling layers (2) 2x2
        # parameters: (kernel size)
        self.avgP1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgP2 = nn.AvgPool2d(kernel_size=(2, 2))
        # Fully connected layers (3)
        # parameters: (input size, output size)
        self.fc1 = nn.Linear(1600, 500) #1024, 512  2952, 1476
        self.fc2 = nn.Linear(500, 128) #1476, 738
        self.fc3 = nn.Linear(128, 10) #738, 369

    def forward(self, x): # Linking all the layers together
        x = self.conv1(x)
        x = torch.sigmoid(x) #activation function
        x = self.avgP1(x)
        x = self.conv2(x)
        x = torch.sigmoid(x) #activation function
        x = self.avgP2(x)
        x = torch.flatten(x, 1) # need to flatten the array before the fully connected layers
        x = self.fc1(x)
        x = torch.sigmoid(x) #activation function
        x = self.fc2(x)
        x = torch.sigmoid(x) #activation function
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1 ) # used for probability training
        return output