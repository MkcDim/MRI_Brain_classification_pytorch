import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


work_path = os.getcwd()

data_path = os.path.join(work_path, 'DATASET2')  

train_path = os.path.join(data_path, 'Training')
test_path = os.path.join(data_path, 'Testing')


# Data augmentation and pre-processing using pytorch

train_transforms = transforms.Compose([transforms.Resize((224, 224)), # Resize the image to 224x224
                                        transforms.RandomRotation(5),    # Random rotation of the image by 5 degrees
                                        transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally
                                        transforms.ToTensor(), # Convert the image to a tensor
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize the image
]) 

#Load the training data

train_data = datasets.ImageFolder(root = train_path, transform=train_transforms)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

print(train_data.class_to_idx)


model = models.resnet50(weights=ResNet50_Weights.DEFAULT)  # Load a pre-trained ResNet model
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)  # Modify the output layer for 4 classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
model.train()

EPOCHS = 100

for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % EPOCHS == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0


# Calculate the accuracy of the model on the training data
model.eval()

#intialize the counters for accuracy calculation
correct = 0
total = 0

with torch.no_grad():
    for data in train_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        #make predictions
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # get the class with the highest probability

        #update the counters
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


# Calculate the accuracy
accuracy = 100 * correct / total
# Print the accuracy
print('Accuracy of the model on the train images: %d %%' % accuracy)


# Load the test data

test_data = datasets.ImageFolder(root = test_path, transform=train_transforms)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)



# Calculate the accuracy of the model on testing data 
model.eval()

#intialize the counters for accuracy calculation
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        #make predictions
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # get the class with the highest probability

        #update the counters
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


# Calculate the accuracy
accuracy = 100 * correct / total
# Print the accuracy
print('Accuracy of the model on the test images: %d %%' % accuracy)

#Predict the class of an image

class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def predict_image(image_path):
    image = Image.open(image_path)

    # Apply the transformation
    image = train_transforms(image).unsqueeze(0)
    model.to(device)
    image = image.to(device)

    # Make a prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    # Get the class label
    class_label = class_names[predicted.item()]
    return  class_label



image_path = os.path.join(data_path,'image(1).jpg')
prediction = predict_image(image_path)
print(f"This is the image of {prediction}.")