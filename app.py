from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('plant_disease_model.h5')

# Define class names for prediction results
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# Upload folder setup
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        if 'plant_image' in request.files:
            file = request.files['plant_image']
            if file.filename != '':
                # Save the uploaded file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                image_path = file_path
                print(f"[INFO] File saved to: {file_path}")

                # Read and preprocess the image
                image = cv2.imread(file_path)
                if image is None:
                    raise FileNotFoundError(f"cv2 could not read the image at path: {file_path}")

                image_resized = cv2.resize(image, (256, 256))
                image_normalized = image_resized / 255.0
                image_input = np.reshape(image_normalized, (1, 256, 256, 3))

                # Predict the disease
                Y_pred = model.predict(image_input)
                result = CLASS_NAMES[np.argmax(Y_pred)]
                plant, disease = result.split('-')
                prediction = f"This is a <strong>{plant}</strong> leaf with <strong>{disease}</strong>."

    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
