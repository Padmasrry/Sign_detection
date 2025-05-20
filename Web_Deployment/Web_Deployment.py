import os
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import pickle
import numpy as np

# Set up Flask app
app = Flask(__name__)
app.template_folder = "Templates"

# Define paths for file uploads and processed files
UPLOAD_FOLDER = "Templates/uploads"
PROCESSED_FOLDER = "Templates/result"

# Ensure the directories exist, if not, create them
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for file uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the trained model
try:
    model_dict = pickle.load(open("../model.p", 'rb'))
    model = model_dict['model']
except Exception as e:
    print(f"Error loading model: {e}")

# Labels dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2',
    29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: 'I love You', 37: 'yes', 38: 'No', 39: 'Hello', 40: 'Thanks',
    41: 'Sorry', 43: 'space'
} 

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to detect hands using skin color and contours
def detect_hand(image):
    # Convert to HSV color space for skin detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Get the largest contour (assumed to be the hand)
    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < 500:
        return None

    # Get bounding box of the hand
    x, y, w, h = cv2.boundingRect(max_contour)
    img_height, img_width = image.shape[:2]
    normalized_features = [x/img_width, y/img_height, (x+w)/img_width, (y+h)/img_height]
    return normalized_features

# Function to process image and return prediction
def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, "Error loading image."

    # Detect hand and extract features
    features = detect_hand(img)
    if features is None:
        return None, "No hands detected."

    # Make a prediction using the model
    prediction = model.predict([np.asarray(features)])
    predicted_index = int(prediction[0])
    predicted_label = labels_dict.get(predicted_index, "Unknown label")

    # Draw the label on the image
    cv2.putText(img, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)

    # Save the processed image
    processed_image_path = os.path.join(PROCESSED_FOLDER, 'processed_' + os.path.basename(image_path))
    cv2.imwrite(processed_image_path, img)

    return processed_image_path, predicted_label

# Route to upload an image
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the image and return the label
            processed_image_path, label = process_image(file_path)
            if processed_image_path:
                return render_template('result.html', label=label, image_path='/processed/' + os.path.basename(processed_image_path))

    return render_template('upload.html')

# Route to serve processed images
@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=8000)