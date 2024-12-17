# Use an official Python image
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the necessary files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U flask-cors

# Expose port for Flask
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
