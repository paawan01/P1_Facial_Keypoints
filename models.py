## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		
		## TODO: Define all the layers of this CNN, the only requirements are:
		## 1. This network takes in a square (same width and height), grayscale image as input
		## 2. It ends with a linear layer that represents the keypoints
		## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
		
		# As an example, you've been given a convolutional layer, which you may (but don't have to) change:
		# 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
		
		## Size of output is determined by formula (W-F+2P)/S + 1
		## W - input size; F - kernel size; P - Padding; S - stride
		
		self.conv1 = nn.Conv2d(1, 32, 5)  # output size : (224-5+0)/1 + 1 = 220 
		self.conv_drop1 = nn.Dropout2d(0.3)

		# Max pooling will make size 110 X 110 
		self.conv2 = nn.Conv2d(32, 64, 3) # output size : (110-3+0)/1 + 1 = 108 
		self.conv_drop2 = nn.Dropout2d(0.3)

		# Max pooling will make size 54 X 54 
		self.conv3 = nn.Conv2d(64, 128, 3) # output size : (54-3+0)/1 + 1 = 52 
		self.conv_drop3 = nn.Dropout2d(0.4)

		# Max pooling will make size 26 X 26 
		self.conv4 = nn.Conv2d(128, 256, 3) # output size : (26-3+0)/1 + 1 = 24 
		self.conv_drop4 = nn.Dropout2d(0.5)

		# Max pooling will make size 12 X 12

		self.fc1 = nn.Linear(12*12*256, 1024)   # Output 
		self.fc_drop1 = nn.Dropout2d(0.5)
		
		self.fc2 = nn.Linear(1024, 1024)
		self.fc_drop2 = nn.Dropout2d(0.6) 

		self.fc3 = nn.Linear(1024, 136)


		self.maxpooling = nn.MaxPool2d(2, 2)



		## Note that among the layers to add, consider including:
		# maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
		

		
	def forward(self, x):
		## TODO: Define the feedforward behavior of this model
		## x is the input image and, as an example, here you may choose to include a pool/conv step:
		## x = self.pool(F.relu(self.conv1(x)))
	   
		x = self.maxpooling(F.relu(self.conv1(x)))
		x = self.conv_drop1(x)
		
		x = self.maxpooling(F.relu(self.conv2(x)))
		x = self.conv_drop2(x)
		
		x = self.maxpooling(F.relu(self.conv3(x)))
		x = self.conv_drop3(x)
		
		x = self.maxpooling(F.relu(self.conv4(x)))
		x = self.conv_drop4(x)
		
		x = x.view(x.size(0), -1)
		
		x = F.relu(self.fc1(x))
		x = self.fc_drop1(x)
		
		x = F.relu(self.fc2(x))
		x = self.fc_drop2(x)
		
		x = self.fc3(x) 


		
		# a modified x, having gone through all the layers of your model, should be returned
		return x
