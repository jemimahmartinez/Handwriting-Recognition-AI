import torch.nn as nn
import torch.nn.functional as F
import torch

# The network should inherit from the nn.Module
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        
        # Convolutional layers (5) 3x3
        # 1: input channals 32: output channels, 3: kernel size, 1: stride
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1)

        # Max Pooling
        # parameters: (kernel size)
        self.maxP1 = nn.MaxPool2d(3)
        self.maxP2 = nn.MaxPool2d(3)
