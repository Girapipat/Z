import tensorflow as tf
import os

def convert_model_to_tflite(h5_path, output_path):
    if not os.path.exists(h5_path):
        print(f"❌ ไม่พบไฟล์: {h5_path}")
        return

    try:
        model = tf.keras.models.load_model(h5_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Optional: เปิด optimization เพื่อลดขนาดไฟล์
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"✅ แปลง {h5_path} เป็น {output_path} สำเร็จแล้ว")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดขณะแปลง {h5_path}: {e}")

if __name__ == "__main__":
    # ระบุ path ของโมเดล .h5 ที่ต้องการแปลง
    classifier_h5 = "solution_classifier.h5"
    regression_h5 = "regression_model.h5"  # เปลี่ยนตามชื่อไฟล์จริงของคุณ

    # ระบุชื่อไฟล์ output .tflite
    classifier_tflite = "solution_classifier.tflite"
    regression_tflite = "regression_model.tflite"

    # แปลงทีละโมเดล
    convert_model_to_tflite(classifier_h5, classifier_tflite)
    convert_model_to_tflite(regression_h5, regression_tflite)
