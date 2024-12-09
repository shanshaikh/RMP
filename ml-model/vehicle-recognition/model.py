# data_preprocessing_and_model_setup.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from PIL import Image
import time

# Step 1: Define Image Transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),       # Resize image to 224x224
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(15),      # Random rotation ±15 degrees
    transforms.ToTensor(),              # Convert to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize (ImageNet stats)
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),      # Resize image to 224x224
    transforms.ToTensor(),              # Convert to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize
])

# Step 2: Load Dataset Using ImageFolder
def load_datasets(train_dir, test_dir):
    train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
    test_dataset = ImageFolder(root=test_dir, transform=test_transforms)
    return train_dataset, test_dataset

# Step 3: DataLoader for Batching
def create_dataloaders(train_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Step 4: Visualize Data
def visualize_sample(data_loader):
    # Get one batch of images and labels
    images, labels = next(iter(data_loader))
    
    # Plot first image in the batch
    plt.figure(figsize=(6, 6))
    plt.imshow(images[0].permute(1, 2, 0))  # Convert (C, H, W) to (H, W, C)
    plt.title(f'Label: {labels[0]}')
    plt.axis('off')
    plt.show()

# Step 5: Load Pre-trained ResNet-50 Model
def initialize_model(num_classes):
    # Load pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    
    # Modify the final layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    return model, criterion, optimizer, device

# Step 6: Train the Model
def train_model(model, criterion, optimizer, train_loader, test_loader, device, num_epochs=10):
    best_model_weights = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Evaluation phase
        model.eval()
        test_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_running_corrects += torch.sum(preds == labels.data)

        test_acc = test_running_corrects.double() / len(test_loader.dataset)

        print(f'Test Acc: {test_acc:.4f}')

        # Deep copy the best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_weights = model.state_dict()

    print(f'Best Test Accuracy: {best_acc:.4f}')
    model.load_state_dict(best_model_weights)
    return model

# Main Function
if __name__ == "__main__":
    # Paths to your train and test datasets
    train_dir = "/Users/shanawazeshaik/myrepo/ml-models/vehicle-recognition/train"
    test_dir = "/Users/shanawazeshaik/myrepo/ml-models/vehicle-recognition/test"
    
    # Load datasets
    train_dataset, test_dataset = load_datasets(train_dir, test_dir)
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset)
    
    # Print dataset sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    
    # Visualize a sample
    visualize_sample(train_loader)
    
    # Initialize model
    num_classes = len(train_dataset.classes)  # Automatically infer number of classes
    model, criterion, optimizer, device = initialize_model(num_classes)
    
    # Train the model
    num_epochs = 10
    trained_model = train_model(model, criterion, optimizer, train_loader, test_loader, device, num_epochs)
    
    # Save the model
    torch.save(trained_model.state_dict(), 'vehicle_recognition_model.pth')
    print("Model training completed and saved as 'vehicle_recognition_model.pth'")
