## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # output size = (W - F + 2P)/S + 1

        # input = (1, 224, 224)
        self.conv1 = nn.Conv2d(1, 16, 5)     # output = (16, 220, 220)  ...(224 - 5 + 2*0)/1 + 1 = 220
        self.pool1 = nn.MaxPool2d(2,2)       # output = (16, 110, 110)  ...(220 - 2 + 2*0)/2 + 1 = 110 = 220//2
        self.drop1 = nn.Dropout(p=0.2)       
        
        self.conv2 = nn.Conv2d(16, 32, 3)    # output = (32, 54, 54)    ...(110 - 3 + 2*0)/1 + 1 = 108
        self.pool2 = nn.MaxPool2d(2,2)       # output = (32, 27, 27)    ...(108 - 2 + 2*0)/2 + 1 = 54 = 108//2
        self.drop2 = nn.Dropout(p=0.3)    
        
        self.conv3 = nn.Conv2d(32, 64, 3)    # output = (64, 52, 52)    ...(54 - 3 + 2*0)/1 + 1 = 52
        self.pool3 = nn.MaxPool2d(2,2)       # output = (64, 26, 26)    ...52//2 = 26
        self.drop3 = nn.Dropout(p=0.4)
        
        # dense layers
        self.fc1 = nn.Linear(64*26*26, 1024)
        self.fc1_drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 136)
        
        
    def forward(self, x):
        
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))
        x = x.view(x.size(0), -1) # flatten before the dense layers
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
    
        return x