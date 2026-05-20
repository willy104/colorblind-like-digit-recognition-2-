#  數字彩色 背景黑白
import math
import random
import os
from PIL import Image, ImageDraw

# PyTorch
import torch
from torchvision import datasets, transforms

try:
    from scipy.spatial import cKDTree as KDTree
    import numpy as np
    IMPORTED_SCIPY = True
except ImportError:
    import numpy as np
    IMPORTED_SCIPY = False


NUM_IMAGES = 5  # 想生成多少張圖片

# 圖片設定
IMAGE_SIZE = 128
TOTAL_CIRCLES = 800
BACKGROUND = (255, 255, 255)

# 顏色設定
color = lambda c: ((c >> 16) & 255, (c >> 8) & 255, c & 255)
COLORS_ON = [color(0x000000),color(0x3C3C3C),color(0x5B5B5B),color(0x7B7B7B),color(0x9D9D9D),color(0xADADAD),color(0xE0E0E0),color(0xFCFCFC)]
COLORS_OFF = [color(0xFF0000),color(0xFF60AF),color(0xFF44FF),color(0xB15BFF),color(0x6A6AFF),color(0x2894FF),color(0x00FFFF),color(0x1AFD9C),color(0x28FF28),color(0x9AFF02),color(0xFFFF37),color(0xFFDC35),color(0xFF8000),color(0xFF5809),color(0xB87070),color(0xAFAF61),color(0x6FB7B7),color(0x9999CC),color(0xB766AD)
]

# ---------- MNIST (PyTorch) ----------
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


def generate_circle(image_width, image_height, min_diameter, max_diameter):
    radius = random.triangular(
        min_diameter, max_diameter,
        max_diameter * 0.8 + min_diameter * 0.2
    ) / 2

    angle = random.uniform(0, math.pi * 2)
    distance_from_center = random.uniform(0, image_width * 0.48 - radius)
    x = image_width * 0.5 + math.cos(angle) * distance_from_center
    y = image_height * 0.5 + math.sin(angle) * distance_from_center

    return x, y, radius


def circle_intersection(circle1, circle2):
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    return (x2 - x1) ** 2 + (y2 - y1) ** 2 < (r2 + r1) ** 2


def circle_draw(draw_image, mnist_image, circle):
    x, y, r = circle
    ix, iy = int(round(x)), int(round(y))

    ix = min(max(ix, 0), mnist_image.width - 1)
    iy = min(max(iy, 0), mnist_image.height - 1)

    pixel = mnist_image.getpixel((ix, iy))
    if pixel[0] < 128:
        fill_color = random.choice(COLORS_ON)
    else:
        fill_color = random.choice(COLORS_OFF)

    draw_image.ellipse(
        (x - r, y - r, x + r, y + r),
        fill=fill_color,
        outline=fill_color
    )


def generater(image_np):
    pil_image = Image.fromarray(image_np).convert('RGB') \
        .resize((256, 256), Image.NEAREST)

    image2 = Image.new('RGB', (256, 256), BACKGROUND)
    draw_image = ImageDraw.Draw(image2)

    width, height = 256, 256
    min_diameter = (width + height) / 200
    max_diameter = (width + height) / 75

    circle = generate_circle(width, height, min_diameter, max_diameter)
    circles = [circle]
    circle_draw(draw_image, pil_image, circle)

    try:
        for _ in range(TOTAL_CIRCLES):
            if IMPORTED_SCIPY:
                kdtree = KDTree([(x, y) for (x, y, _) in circles])
                while True:
                    circle = generate_circle(width, height, min_diameter, max_diameter)
                    elements, indexes = kdtree.query(
                        [(circle[0], circle[1])], k=12
                    )
                    for element, index in zip(elements[0], indexes[0]):
                        if not np.isinf(element) and circle_intersection(circle, circles[index]):
                            break
                    else:
                        break
            else:
                while any(circle_intersection(circle, c) for c in circles):
                    circle = generate_circle(width, height, min_diameter, max_diameter)

            circles.append(circle)
            circle_draw(draw_image, pil_image, circle)

    except KeyboardInterrupt:
        pass

    return image2


# ---------- 顯示 ----------
from IPython.display import display

for idx in range(NUM_IMAGES):
    img, label = train_dataset[idx]

    # Tensor → numpy (28x28)
    img_np = (img.squeeze().numpy() * 255).astype("uint8")

    ish_img = generater(img_np)
    display(ish_img)
