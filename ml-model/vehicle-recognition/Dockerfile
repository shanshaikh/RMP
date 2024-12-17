# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for GUI support
RUN apt-get update && apt-get install -y \
    python3-tk \
    libgl1-mesa-glx \
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt gui_classifier.py /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the required application files
COPY test_images /app/test_images

# Expose a port if your script will run as a server (optional)
# EXPOSE 8080

# Command to run the application
CMD ["python", "gui_classifier.py"]
