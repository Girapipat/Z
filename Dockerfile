# ใช้ base image ของ Python
FROM python:3.10-slim

# ตั้ง working directory
WORKDIR /app

# คัดลอกไฟล์
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# คัดลอกทุกไฟล์ไปไว้ใน container
COPY . .

# รัน Flask app
CMD ["python", "app_ai.py"]
