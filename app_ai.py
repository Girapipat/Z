import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# ---------------------------
# ฟังก์ชันโหลดโมเดล TFLite
# ---------------------------
def load_tflite_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ไม่พบโมเดล: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# ---------------------------
# โหลดโมเดลทั้งสอง
# ---------------------------
classifier_interpreter = load_tflite_model("models/classifier_model.tflite")
regression_interpreter = load_tflite_model("models/regression_model.tflite")

# ---------------------------
# ฟังก์ชันรันโมเดล TFLite
# ---------------------------
def run_tflite_model(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # แปลง input ให้ตรง type
    input_data = input_data.astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# ---------------------------
# เตรียมภาพก่อนส่งเข้าโมเดล
# ---------------------------
def preprocess_image(image, target_size=(224, 224)):
    img = image.convert("RGB").resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)
    return img_array

# ---------------------------
# API อัปโหลดภาพและวิเคราะห์
# ---------------------------
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "กรุณาอัปโหลดไฟล์ภาพ"}), 400

    file = request.files["file"]
    try:
        image = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "ไฟล์ที่อัปโหลดไม่ใช่ภาพ"}), 400

    # เตรียม input
    input_data = preprocess_image(image)

    # --------------------
    # 1) Predict Classifier
    # --------------------
    class_output = run_tflite_model(classifier_interpreter, input_data)
    # สมมติว่า output = [[probability_of_solution]]
    is_solution_prob = float(class_output[0][0])
    is_solution = is_solution_prob >= 0.5

    if not is_solution:
        return jsonify({
            "is_solution": False,
            "confidence": is_solution_prob
        })

    # --------------------
    # 2) Predict Regression
    # --------------------
    reg_output = run_tflite_model(regression_interpreter, input_data)
    intensity_value = float(reg_output[0][0])  # ค่าความเข้ม

    return jsonify({
        "is_solution": True,
        "confidence": is_solution_prob,
        "intensity": intensity_value
    })

# ---------------------------
# Run Flask App
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
