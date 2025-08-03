import os
import random
from PIL import Image, ImageDraw

SAVE_DIR = "dataset/not_solution"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
NUM_IMAGES = 200

def generate_random_image(path):
    img = Image.new("RGB", IMG_SIZE, color=random_color())
    draw = ImageDraw.Draw(img)

    for _ in range(random.randint(5, 20)):
        shape_type = random.choice(["rectangle", "ellipse", "line"])
        x1, y1 = random.randint(0, 150), random.randint(0, 150)
        x2, y2 = x1 + random.randint(20, 70), y1 + random.randint(20, 70)
        color = random_color()

        if shape_type == "rectangle":
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        elif shape_type == "ellipse":
            draw.ellipse([x1, y1, x2, y2], outline=color, width=2)
        elif shape_type == "line":
            draw.line([x1, y1, x2, y2], fill=color, width=2)

    img.save(path)

def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

for i in range(NUM_IMAGES):
    filepath = os.path.join(SAVE_DIR, f"not_solution_{i+1:03d}.png")
    generate_random_image(filepath)

print(f"✅ สร้างภาพ not_solution จำนวน {NUM_IMAGES} รูปไว้ที่ '{SAVE_DIR}' แล้ว")
