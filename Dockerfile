# ใช้ Python 3.11 ที่ Render รองรับ และ TensorFlow ยังมี wheel
FROM python:3.11-slim

# ป้องกันปัญหา interactive
ENV PYTHONUNBUFFERED=1

# ติดตั้งระบบพื้นฐานที่ TensorFlow ต้องใช้
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# คัดลอกไฟล์ requirements.txt
COPY requirements.txt .

# อัปเกรด pip และติดตั้ง dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# คัดลอกโค้ดทั้งหมด
COPY . .

# รันแอป (ถ้าใช้ Flask หรือ FastAPI ให้เปลี่ยนคำสั่งตามนั้น)
CMD ["python", "app_ai.py"]
