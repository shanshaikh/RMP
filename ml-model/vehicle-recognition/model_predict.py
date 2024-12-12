### Imports ###
import torch
import boto3
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import os
from io import BytesIO

# Function to load the trained model
def load_model(model_path, num_classes):
    # Recreate the ResNet-50 architecture
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to preprocess the image
def preprocess_image(image_path):
    # Define transformations (match training transformations)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),         # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])

    # Open the image
    image = Image.open(image_path).convert('RGB')  # Ensure RGB format
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to make predictions
def predict_image(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)  # Perform forward pass
        _, predicted = torch.max(outputs, 1)  # Get the class index with highest score
        return class_names[predicted.item()]  # Map index to class name

## Get trained model from S3 ##
def get_model_from_s3(bucket_name, object_name, num_classes):
    """
    Retrieve a trained model directly from S3 into memory and load it into a PyTorch model.

    Args:
        bucket_name (str): Name of the S3 bucket.
        object_name (str): Key of the object in S3.
        num_classes (int): Number of output classes for the model.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    s3_client = boto3.client('s3')
    try:
        # Get the object from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
        model_data = response['Body'].read()  # Read the file content

        # Load the state_dict with CPU mapping
        state_dict = torch.load(BytesIO(model_data), map_location=torch.device('cpu'), weights_only=True)  # Load directly from BytesIO

        # Load the model architecture
        model = models.resnet50()  # Match your training architecture
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        model.load_state_dict(state_dict)
        model.eval()  # Set the model to evaluation mode

        print("Model successfully loaded from S3 and ready for inference!")
        return model
    except Exception as e:
        print(f"Error retrieving or loading model: {e}")
        return None

# Main script for prediction
if __name__ == "__main__":
    # Paths and settings
    #model_path = 'drive/MyDrive/Colab Notebooks/vehicles/vehicle_recognition_model.pth'  # Path to your saved model
    images_folder = 'test_images'        # Replace with the path to your image
    class_names = ['Auto Rickshaw', 'Bicycle', 'Car', 'Motorcycle', 'Plane', 'Ship', 'Train']  # Replace with your actual class names

    # Load the model
    bucket_name = "shaikmlmodeltest"
    object_name = "models/vehicle_recognition_model.pth"
    num_classes = len(class_names)
    model = get_model_from_s3(bucket_name, object_name, num_classes)
    
    # Iterate through the folder for all images
    for image in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image)
        
        if image.endswith(('.jpeg', '.jpg', '.png')):
            # Preprocess the input image
            image_tensor = preprocess_image(image_path)

            # Predict the class of the image
            predicted_class = predict_image(model, image_tensor, class_names)

            # Output the prediction
            print(f"\nI know what that is, that's a {predicted_class}!")