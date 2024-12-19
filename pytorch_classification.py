import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import os


work_path = os.getcwd()

data_path = os.path.join(work_path, 'DATASET2')  

train_path = os.path.join(data_path, 'Training')
test_path = os.path.join(data_path, 'Testing')


# Data augmentation and pre-processing using pytorch

train_transforms = transforms.Compose([transforms.RandomRotation(5),    # Random rotation of the image by 5 degrees
                                        transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally
                                        transforms.ToTensor(), # Convert the image to a tensor
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize the image
]) 

#Load the training data

train_data = datasets.ImageFolder(root = train_path, transform=train_transforms)


