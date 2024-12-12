# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the required application files
COPY model_predict.py /app/
COPY test_images/ /app/test_images/

# Expose a port if your script will run as a server (optional)
# EXPOSE 8080

# Command to run the application
CMD ["python", "model_predict.py"]
