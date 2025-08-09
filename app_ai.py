import tensorflow as tf
import numpy as np
from PIL import Image

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
# ---------------------------
def preprocess_image(image_path, target_size):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, h, w, 3)
    return img_array


# ---------------------------
# ฟังก์ชันทำนายด้วย classifier
# ---------------------------
def predict_classifier(image_path):
    img_array = preprocess_image(image_path, (224, 224))
    interpreter_cls.set_tensor(input_details_cls[0]['index'], img_array)
    interpreter_cls.invoke()
    pred = interpreter_cls.get_tensor(output_details_cls[0]['index'])[0][0]
    label = "solution" if pred >= 0.5 else "not_solution"
    return label, float(pred)


# ---------------------------
# ฟังก์ชันทำนายด้วย regression
# ---------------------------
def predict_regression(image_path):
    img_array = preprocess_image(image_path, (224, 224))
    interpreter_reg.set_tensor(input_details_reg[0]['index'], img_array)
    interpreter_reg.invoke()
    pred = interpreter_reg.get_tensor(output_details_reg[0]['index'])[0][0]
    return float(pred)  # ค่าระหว่าง 0.0 - 1.0


# ---------------------------
# ตัวอย่างการใช้งาน
# ---------------------------
if __name__ == "__main__":
    test_image = "test.jpg"  # เปลี่ยนเป็น path ของภาพที่จะทดสอบ

    label, score = predict_classifier(test_image)
    print(f"Classifier: {label} (score={score:.4f})")

    intensity = predict_regression(test_image)
    print(f"Regression intensity: {intensity:.4f}")
