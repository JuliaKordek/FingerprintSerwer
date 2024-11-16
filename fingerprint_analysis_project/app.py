
from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Directory for uploaded files
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Function for thinning using OpenCV's ximgproc
def thin_image_opencv(binary_image):
    if binary_image.dtype != np.uint8:
        binary_image = binary_image.astype(np.uint8)
    thinned = cv2.ximgproc.thinning(binary_image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return thinned

# Route for the homepage
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the file was uploaded
        if 'file' not in request.files:
            return "No file uploaded.", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected.", 400

        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the fingerprint
        processed_file_path = process_fingerprint(file_path)
        return redirect(url_for("results", filename=os.path.basename(processed_file_path)))

    return render_template("index.html")

# Route to display results
@app.route("/results/<filename>")
def results(filename):
    processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    return render_template("results.html", filename=filename, processed_file=processed_file_path)

# Route to serve processed files
@app.route("/processed/<filename>")
def processed_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename))

# Function to process the fingerprint image
def process_fingerprint(file_path):
    try:
        # Load the image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Invalid image.")

        # Preprocess the image
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thinned_image = thin_image_opencv(binary_image)

        # Save the processed result
        processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(file_path))
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Binary Image")
        plt.imshow(binary_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Thinned Image")
        plt.imshow(thinned_image, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(processed_file_path)
        plt.close()

        return processed_file_path
    except Exception as e:
        raise RuntimeError(f"Failed to process fingerprint: {e}")

if __name__ == "__main__":
    app.run(debug=True)
