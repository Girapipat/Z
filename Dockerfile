# ใช้ Python เวอร์ชันที่ Render รองรับ
FROM python:3.10-slim

# ตั้งค่าให้ Python แสดง log แบบไม่ buffer
ENV PYTHONUNBUFFERED=1

# สร้างโฟลเดอร์ app
WORKDIR /app

# ติดตั้ง dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# คัดลอกโค้ดทั้งหมดไปใน container
COPY . .

# Port สำหรับ web server
EXPOSE 8000

# คำสั่งรันแอป (แก้ตาม framework ของคุณ)
# ถ้าใช้ Flask:
# CMD ["python", "app_ai.py"]

# ถ้าใช้ FastAPI + Uvicorn:
CMD ["uvicorn", "app_ai:app", "--host", "0.0.0.0", "--port", "8000"]
