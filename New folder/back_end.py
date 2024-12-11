from flask import Flask, request, jsonify
import os

# Initialize Flask app
app = Flask(__name__)

# Ensure upload directory exists
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Read raw binary data from request body
        image_data = request.data
        
        # Validate the content length
        if not image_data:
            return jsonify({"error": "No data in request body"}), 400
        
        # Save the image to a file
        file_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.jpg')  # Save as "uploaded_image.jpg"
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        return jsonify({"message": "Image received and saved", "file_path": file_path}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == '__main__':
    app.run(debug=True, port=8090)
