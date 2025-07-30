# 📁 train_all.py
import train_classifier
import train_regression
import os

DATASET_DIR = r"C:\Users\seapr\Pictures\งาน\ai-solution-classifier_No.9\dataset"

print("\n--- Training classifier model ---")
try:
    binary_data_dir = train_classifier.simplify_class_folder_structure(DATASET_DIR)
    print("✅ binary dataset:", binary_data_dir)
    train_classifier.train_classifier(binary_data_dir)  # <== ส่ง path ที่จัดเรียบร้อยแล้ว
except Exception as e:
    print("❌ เกิดข้อผิดพลาดในการเทรน classifier:", e)

print("\n--- Training regression model ---")
try:
    train_regression.train_regression(DATASET_DIR)
except Exception as e:
    print("❌ เกิดข้อผิดพลาดในการเทรน regression:", e)
