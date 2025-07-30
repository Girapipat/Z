### ✅ app_ai.py

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load models
model_class = tf.keras.models.load_model("classifier_model.h5")
model_reg = tf.keras.models.load_model("regression_model.h5", compile=False)

IMG_SIZE = (224, 224)

# Prepare image

def prepare_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "ไม่พบไฟล์ภาพ"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "ยังไม่ได้เลือกไฟล์"})

    img = prepare_image(file)

    # STEP 1: Classification
    pred_prob = model_class.predict(img)[0][0]
    threshold = 0.5

    if pred_prob < threshold:
        return jsonify({"is_solution": False, "confidence": float(pred_prob)})

    # STEP 2: Regression
    pred_value = model_reg.predict(img)[0][0]
    intensity = int(pred_value * 255)

    return jsonify({
        "is_solution": True,
        "confidence": float(pred_prob),
        "intensity": intensity
    })

if __name__ == "__main__":
    app.run(debug=True)