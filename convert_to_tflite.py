# convert_models_to_tflite.py
import tensorflow as tf

def convert_h5_to_tflite(h5_path, tflite_path):
    print(f"🔄 กำลังแปลง {h5_path} → {tflite_path} ...")
    model = tf.keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # เปิดการ Optimize เพื่อลดขนาดไฟล์
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ แปลงเสร็จ: {tflite_path}")

if __name__ == "__main__":
    # แปลง classifier
    convert_h5_to_tflite(
        h5_path="classifier_model.h5",
        tflite_path="classifier_model.tflite"
    )

    # แปลง regression
    convert_h5_to_tflite(
        h5_path="regression_model.h5",
        tflite_path="regression_model.tflite"
    )
