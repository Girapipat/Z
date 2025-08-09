import os
import numpy as np
from flask import Flask, request, render_template
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# โหลดโมเดล TFLite
def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

# โหลดทั้ง 2 โมเดล
cls_interpreter, cls_input_details, cls_output_details = load_tflite_model("classifier_model.tflite")
reg_interpreter, reg_input_details, reg_output_details = load_tflite_model("regression_model.tflite")

# ฟังก์ชันประมวลผลภาพให้เหมาะกับ MobileNetV2
def preprocess_image(image):
    image = image.resize((224, 224))
    image = image.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ฟังก์ชันทำนายว่าเป็นสารละลายหรือไม่
def is_solution(image_array):
    cls_interpreter.set_tensor(cls_input_details[0]['index'], image_array)
    cls_interpreter.invoke()
    output_data = cls_interpreter.get_tensor(cls_output_details[0]['index'])
    result = np.argmax(output_data)
    return result == 1  # 1 คือ solution, 0 คือ not_solution

# ฟังก์ชันทำนายค่าความเข้มข้น (เฉพาะถ้าเป็นสารละลาย)
def predict_intensity(image_array):
    reg_interpreter.set_tensor(reg_input_details[0]['index'], image_array)
    reg_interpreter.invoke()
    output_data = reg_interpreter.get_tensor(reg_output_details[0]['index'])
    value = output_data[0][0]  # ค่าอยู่ในช่วง 0.0–0.9
    intensity = int(value * 255)  # แปลงเป็น 0–255
    return intensity

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file.stream)
            img_array = preprocess_image(image)

            if is_solution(img_array):
                intensity = predict_intensity(img_array)
                prediction = f"เป็นสารละลาย ความเข้มข้น: {intensity}"
            else:
                prediction = "ไม่ใช่สารละลาย"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Render จะส่ง PORT เข้ามา
    app.run(host='0.0.0.0', port=port)
