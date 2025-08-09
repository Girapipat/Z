import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, render_template

app = Flask(__name__)

# ---------------------------
# โหลด .tflite ทั้ง 2 โมเดล
# ---------------------------
# Classifier
interpreter_cls = tf.lite.Interpreter(model_path="classifier_model.tflite")
interpreter_cls.allocate_tensors()
input_details_cls = interpreter_cls.get_input_details()
output_details_cls = interpreter_cls.get_output_details()

# Regression
interpreter_reg = tf.lite.Interpreter(model_path="regression_model.tflite")
interpreter_reg.allocate_tensors()
input_details_reg = interpreter_reg.get_input_details()
output_details_reg = interpreter_reg.get_output_details()


# ---------------------------
# ฟังก์ชันประมวลผลภาพก่อนส่งเข้าโมเดล
# ปรับให้รับได้ทั้ง path หรือ PIL.Image object
# ---------------------------
def preprocess_image(image, target_size):
    if isinstance(image, str):
        # กรณีรับเป็น path
        img = Image.open(image).convert("RGB")
    else:
        # กรณีรับเป็น PIL.Image
        img = image.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, h, w, 3)
    return img_array


# ---------------------------
# ฟังก์ชันทำนายด้วย classifier
# ---------------------------
def predict_classifier(image):
    img_array = preprocess_image(image, (224, 224))
    interpreter_cls.set_tensor(input_details_cls[0]['index'], img_array)
    interpreter_cls.invoke()
    pred = interpreter_cls.get_tensor(output_details_cls[0]['index'])[0][0]
    label = "solution" if pred >= 0.5 else "not_solution"
    return label, float(pred)


# ---------------------------
# ฟังก์ชันทำนายด้วย regression
# ---------------------------
def predict_regression(image):
    img_array = preprocess_image(image, (224, 224))
    interpreter_reg.set_tensor(input_details_reg[0]['index'], img_array)
    interpreter_reg.invoke()
    pred = interpreter_reg.get_tensor(output_details_reg[0]['index'])[0][0]
    return float(pred)  # ค่าระหว่าง 0.0 - 1.0


# ---------------------------
# Flask route สำหรับรับไฟล์อัปโหลดและทำนายผล
# ---------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        img = Image.open(file.stream)

        label, score = predict_classifier(img)
        intensity = predict_regression(img)

        return f"Label: {label} (score={score:.4f}), Intensity: {intensity:.4f}"

    # แสดงหน้า HTML form สำหรับอัปโหลดไฟล์ (ต้องมีไฟล์ index.html)
    return render_template('index.html')


# ---------------------------
# ตัดส่วนนี้ออกเพราะบน Render ไม่มีไฟล์ test.jpg
# ---------------------------
# if __name__ == "__main__":
#     test_image_path = "test.jpg"
#     label, score = predict_classifier(test_image_path)
#     print(f"Classifier: {label} (score={score:.4f})")
#
#     intensity = predict_regression(test_image_path)
#     print(f"Regression intensity: {intensity:.4f}")


if __name__ == '__main__':
    app.run(debug=True)
