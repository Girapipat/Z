import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------
# CONFIG
# -------------------------
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "classifier_model.h5"
IMG_SIZE = (224, 224)

# -------------------------
# APP INIT
# -------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# โหลดโมเดล
print("🔄 กำลังโหลดโมเดล...")
model = load_model(MODEL_PATH)
print("✅ โหลดโมเดลสำเร็จ")

# -------------------------
# ฟังก์ชันช่วยประมวลผลภาพ
# -------------------------
def prepare_image(file_path):
    img = Image.open(file_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def calculate_intensity(file_path):
    """คำนวณค่าความเข้มข้นจากค่าเฉลี่ยความสว่าง (0-255)"""
    img = Image.open(file_path).convert("L")  # แปลงเป็น grayscale
    return int(np.mean(np.array(img)))

# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "ไม่มีไฟล์ที่อัปโหลด"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "กรุณาเลือกไฟล์"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        # เตรียมภาพและทำนาย
        img_array = prepare_image(file_path)
        pred = model.predict(img_array)[0][0]

        confidence = float(pred if pred > 0.5 else 1 - pred)
        is_solution = bool(pred > 0.5)

        if not is_solution:
            return jsonify({
                "is_solution": False,
                "confidence": confidence
            })

        # คำนวณค่าความเข้มข้น
        intensity = calculate_intensity(file_path)

        return jsonify({
            "is_solution": True,
            "confidence": confidence,
            "intensity": intensity
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
