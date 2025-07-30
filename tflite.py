import tensorflow as tf

# -----------------------------
# 1. แปลง classifier_model.h5
# -----------------------------
classifier_model = tf.keras.models.load_model('classifier_model.h5')
classifier_converter = tf.lite.TFLiteConverter.from_keras_model(classifier_model)
classifier_tflite_model = classifier_converter.convert()

with open('classifier_model.tflite', 'wb') as f:
    f.write(classifier_tflite_model)

print("✅ แปลง classifier_model.h5 เป็น classifier_model.tflite แล้ว")

# -----------------------------
# 2. แปลง regression_model.h5
# -----------------------------
regression_model = tf.keras.models.load_model('regression_model.h5')
regression_converter = tf.lite.TFLiteConverter.from_keras_model(regression_model)
regression_tflite_model = regression_converter.convert()

with open('regression_model.tflite', 'wb') as f:
    f.write(regression_tflite_model)

print("✅ แปลง regression_model.h5 เป็น regression_model.tflite แล้ว")
