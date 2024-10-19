from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import os
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = 'static'

# Load models
models = {}
models['model1'] = load_model("model/forgery_detection_model_Xception.h5")
#models['model2'] = load_model("model/forgery_detection_model.h5")
#models['model3'] = load_model("model3.h5")

def apply_ela(image_path, quality=90):
    image = Image.open(image_path).convert('RGB')
    image.save('temp.jpg', 'JPEG', quality=quality)
    ela_image = Image.open('temp.jpg')
    ela_image = ImageChops.difference(image, ela_image )
    band_values = ela_image.getextrema()
    max_value = max([val[1] for val in band_values])
    if max_value == 0:
        max_value = 1
    scale = 255.0 / max_value
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

def preprocess_image(image_path):
    ela_image = apply_ela(image_path)
    image = ela_image.resize((128, 128))  # Resize to match the input shape of the model
    return np.array(image) / 255.0  # Normalize pixel values

def predict_image(model, image_path):
    preprocessed_image = preprocess_image(image_path)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
    prediction = model.predict(preprocessed_image)[0][0]
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get uploaded file
        file = request.files['file']
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            img = os.path.join(app.config['UPLOAD_FOLDER'], filename)  
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            print("file name:" ,filename)
            print("img: ",img)

            # Perform predictions
            predictions = {}
            for model_name, model in models.items():
                prediction = predict_image(model, filename)
                predictions[model_name] = prediction

            return render_template('home.html', predictions=predictions, img=img)
    return render_template('home.html', predictions=None, file_path=None)

if __name__ == '__main__':
    app.run(debug=True)
