import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import os
import argparse

def data_augmentation(dataset_path):

    # Data augmentation and pre-processing using pytorch
    data_transforms = transforms.Compose([transforms.Resize((224, 224)), # Resize the image to 224x224
                                            transforms.RandomRotation(5),    # Random rotation of the image by 5 degrees
                                            transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally
                                            transforms.ToTensor(), # Convert the image to a tensor
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize the image
    ]) 

    #Load the training data

    transformed_data = datasets.ImageFolder(root = dataset_path, transform=data_transforms)   
    transformed_data_loader = DataLoader(transformed_data, batch_size=32, shuffle=True)
    print(transformed_data.class_to_idx)

    return transformed_data_loader , data_transforms

def model_initialisation(device, learning_rate = 0.001, nb_classes = 4):

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)  # Load a pre-trained ResNet model
    #model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, nb_classes)  # Modify the output layer for 4 classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= learning_rate)

    return model, criterion, optimizer

def training_model(device, model, criterion, optimizer, train_loader, epochs):
    
    
    model.train()
    for epoch in range(epochs):
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
            if i % epochs == 9:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
    print('Finished Training')

    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model

def model_evaluation(device, model, data_loader, type_loader):

    # Calculate the accuracy of the model on the dataset
    model.eval()


    #intialize the counters for accuracy calculation
    correct = 0
    total = 0

    with torch.no_grad():
        for data in data_loader:
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
    print('Accuracy of the model on the ' + type_loader + ' images: %d %%' % accuracy)
    
    
# Function to predict the class of an image
def predict_image(device, model, image_path,class_names,data_transforms):
    image = Image.open(image_path)

    # Apply the transformation
    image = data_transforms(image).unsqueeze(0)
    model.to(device)
    image = image.to(device)

    # Make a prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

        # Get the class label
        class_label = class_names[predicted.item()]
    return  class_label

def main():

    parser = argparse.ArgumentParser(description="Predict the class of an MRI brain cancer image")
    parser.add_argument('--data_root_dir', type=str, default='./DATASET2', 
                    help='data directory')
    parser.add_argument('--max_epochs', type=int, default=10,
                    help='maximum number of epochs to train (default: 10)')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    data_path = args.data_root_dir

    train_path = os.path.join(data_path, 'Training')
    test_path = os.path.join(data_path, 'Testing')

    train_transformed_loader , data_transforms = data_augmentation(train_path)
    test_transformed_loader , _ = data_augmentation(test_path)

    model, criterion, optimizer = model_initialisation(device, learning_rate = 0.001, nb_classes = 4)

    if os.path.exists(os.path.join(args.results_dir, 'model.pth')):
        model = load_model(model, os.path.join(args.results_dir, 'model.pth'))
    else : 
        trained_model = training_model(device, model, criterion, optimizer, train_transformed_loader, args.max_epochs)
        save_model(trained_model, os.path.join(args.results_dir, 'model.pth'))

    model_evaluation(device, model, train_transformed_loader, 'training')

    model_evaluation(device, model, test_transformed_loader, 'testing')

    #Predict the class of an image

    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

    image_path = os.path.join(data_path,'image(1).jpg')
    prediction = predict_image(device, model, image_path,class_names,data_transforms)
    print(f"This is the image of {prediction}.")


if __name__ == "__main__":
    main()
    print("finished!")
    print("end script")