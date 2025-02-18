from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
from io import BytesIO
import boto3

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend communication

# Define your model loading function
def get_model_from_s3(bucket_name, object_name, num_classes):
    s3_client = boto3.client('s3')
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
        model_data = response['Body'].read()  # Read the file content
        state_dict = torch.load(BytesIO(model_data), map_location=torch.device('cpu'))
        model = models.resnet50()  
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"Error retrieving or loading model: {e}")
        return None

# Preprocess the uploaded image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Predict the image class
def predict_image(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# Predict the image class
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if an image file was uploaded
        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        # Get the uploaded image
        file = request.files["image"]
        image = Image.open(BytesIO(file.read())).convert('RGB')

        # Load the model from S3 (Replace with your S3 details)
        bucket_name = "shaikmlmodeltest"
        object_name = "models/vehicle_recognition_model.pth"
        class_names = ['Auto Rickshaw', 'Bicycle', 'Car', 'Motorcycle', 'Plane', 'Ship', 'Train']
        model = get_model_from_s3(bucket_name, object_name, len(class_names))

        if model is None:
            return jsonify({"error": "Error loading the model"}), 500

        # Preprocess the image and predict
        image_tensor = preprocess_image(image)
        predicted_class = predict_image(model, image_tensor, class_names)

        # Return the prediction as JSON
        return jsonify({"prediction": predicted_class}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
