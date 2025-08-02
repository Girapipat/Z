from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
IMG_SIZE = (224, 224)

def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

# Load TFLite models
model_class = load_tflite_model("classifier_model.tflite")
model_reg = load_tflite_model("regression_model.tflite")

input_class = model_class.get_input_details()
output_class = model_class.get_output_details()
input_reg = model_reg.get_input_details()
output_reg = model_reg.get_output_details()

def prepare_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
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
    model_class.set_tensor(input_class[0]['index'], img)
    model_class.invoke()
    pred_prob = model_class.get_tensor(output_class[0]['index'])[0][0]
    threshold = 0.5

    if pred_prob < threshold:
        return jsonify({"is_solution": False, "confidence": float(pred_prob)})

    # STEP 2: Regression
    model_reg.set_tensor(input_reg[0]['index'], img)
    model_reg.invoke()
    pred_value = model_reg.get_tensor(output_reg[0]['index'])[0][0]
    intensity = int(pred_value * 255)

    return jsonify({
        "is_solution": True,
        "confidence": float(pred_prob),
        "intensity": intensity
    })

if __name__ == "__main__":
    app.run(debug=True)
