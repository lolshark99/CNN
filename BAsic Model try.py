import numpy as np
import torch
import cv2
import torch.nn as nn
import kagglehub
import torch.optim as optim
# Download latest version
path = kagglehub.dataset_download("msambare/fer2013")

print("Path to dataset files:", path)

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):   
        super(EmotionCNN , self).__init__()
        self.conv1 = nn.Conv2d(1 , 32 ,kernel_size = 2)
        self.maxpool1 = nn.MaxPool2d(2 , 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32 , 64 , kernel_size = 2)
        self.maxpool2 = nn.MaxPool2d(2 , 2)
        self.dropout = nn.Dropout(0.5)
        self.Fc1 = nn.Linear(11 * 11 * 64 , 128)
        self.Fc2 = nn.Linear(128 , num_classes)

    def forward(self,X):
        X = self.conv1(X)
        X = self.maxpool1(X)
        X = self.conv2(X) 
        X = self.maxpool2(X)
        X = self.relu(X)
        X = self.Fc1(X)
        X = self.Fc2(X)

        return X
model = EmotionCNN()
     
dummy_input = torch.randn(1, 1, 48, 48)

# Forward pass
output = model(dummy_input)
print("Output shape:", output.shape)  # Should be [1, 7] for 7 classes