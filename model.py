"""
Module provides CNN model
"""
from torch.nn import nn
#import torch.nn.functional as F


class CNNModel(nn.Module):
    '''
        The CNN model as a PyTorch class
        Args:
            nn.Module: PyTorch neural network object
        Attributes:
    '''

    def __init__(self, n_classes):
        '''Inits the model
        Args:
            distance (int): The amount of distance traveled
            n_classes (int): 4 directions
        Raises:
            -
        Returns:
            -
        '''
        super(CNNModel, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 37 * 37, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(256, n_classes)

    def forward(self, x):
        '''Forward pass for the model
        Args:
            self: model itself
            x: Input
        Raises:
            -
        Returns:
            -
        '''
        # Forward pass through the network
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 128 * 37 * 37)

        x = self.relu4(self.fc1(x))
        x = self.relu5(self.fc2(x))
        x = self.fc3(x)

        return x
