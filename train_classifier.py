import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

def train_classifier(binary_dataset_dir, output_model_path="solution_classifier.h5", image_size=(224, 224), batch_size=16, epochs=10):
    print(f"เริ่มฝึกโมเดลจาก dataset: {binary_dataset_dir}")

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        binary_dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        binary_dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
    base_model.trainable = False  # Freeze

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(output_model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[checkpoint]
    )

    print(f"✅ ฝึกเสร็จและบันทึกโมเดลไว้ที่: {output_model_path}")

if __name__ == "__main__":
    dataset_path = os.path.join("dataset", "_binary")  # แก้ path ให้ตรงกับโฟลเดอร์จริง
    output_path = "solution_classifier.h5"
    
    try:
        train_classifier(dataset_path, output_model_path=output_path)
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการเทรน classifier: {e}")
