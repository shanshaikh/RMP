import torch
from torchvision import models, transforms
from PIL import Image

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

# Main script for prediction
if __name__ == "__main__":
    # Paths and settings
    model_path = 'vehicle_recognition_model.pth'  # Path to your saved model
    image_path = '<test_image>'        # Replace with the path to your image
    class_names = ['car', 'bike', 'motorcycle', 'rickshaw', 'plane', 'ship', 'train']  # Replace with your actual class names

    # Load the model
    num_classes = len(class_names)
    model = load_model(model_path, num_classes)

    # Preprocess the input image
    image_tensor = preprocess_image(image_path)

    # Predict the class of the image
    predicted_class = predict_image(model, image_tensor, class_names)

    # Output the prediction
    print(f"Predicted class: {predicted_class}")
