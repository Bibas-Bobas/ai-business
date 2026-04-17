import cv2
import numpy as np
import os
import random
from pathlib import Path

INPUT_DIR = 'Dataset_2025_IceClass'  # Папка с твоими исходными фреймами
OUTPUT_DIR = 'dataset'  # Папка, куда сохранится результат
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tif', '.bmp']

# Создаем структуру папок
mask_types = ['horizontal_tri', 'vertical_tri', 'irregular_shape']
for m_type in mask_types:
    for sub in ['masked', 'masks', 'original']:
        os.makedirs(os.path.join(OUTPUT_DIR, m_type, sub), exist_ok=True)


def generate_horizontal_triangle_mask(w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    # Генерируем узкий клин от левого или правого края к центру
    side = random.choice(['left', 'right'])
    y_start = random.randint(0, h // 2)
    y_end = random.randint(h // 2, h)
    thickness = random.randint(h // 10, h // 3)

    if side == 'left':
        pts = np.array([[0, y_start], [w, random.randint(0, h)], [0, y_start + thickness]])
    else:
        pts = np.array([[w, y_start], [0, random.randint(0, h)], [w, y_start + thickness]])

    cv2.fillPoly(mask, [pts], 255)
    return mask


def generate_vertical_triangle_mask(w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    # Генерируем клин сверху или снизу
    side = random.choice(['top', 'bottom'])
    x_start = random.randint(0, w // 2)
    thickness = random.randint(w // 10, w // 3)

    if side == 'top':
        pts = np.array([[x_start, 0], [random.randint(0, w), h], [x_start + thickness, 0]])
    else:
        pts = np.array([[x_start, h], [random.randint(0, w), 0], [x_start + thickness, h]])

    cv2.fillPoly(mask, [pts], 255)
    return mask


def generate_irregular_mask(w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    # Создаем случайный многоугольник от 5 до 8 вершин
    num_pts = random.randint(5, 8)
    pts = []
    for _ in range(num_pts):
        pts.append([random.randint(0, w), random.randint(0, h)])
    pts = np.array(pts, np.int32)
    # Сортируем точки, чтобы полигон не был слишком "самопересекающимся" (опционально)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def process_images():
    image_paths = [p for p in Path(INPUT_DIR).glob('**/*') if p.suffix.lower() in IMG_EXTENSIONS]

    if not image_paths:
        print(f"В папке {INPUT_DIR} не найдено изображений!")
        return

    print(f"Найдено изображений: {len(image_paths)}. Начинаю генерацию...")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None: continue

        h, w = img.shape[:2]
        base_name = img_path.stem

        # Генерация 3 видов масок
        generators = {
            'horizontal_tri': generate_horizontal_triangle_mask,
            'vertical_tri': generate_vertical_triangle_mask,
            'irregular_shape': generate_irregular_mask
        }

        for m_type, func in generators.items():
            # Для каждой картинки делаем по 3 вариации маски одного типа (можно поменять на 1)
            for i in range(1, 4):
                mask = func(w, h)

                # Применяем маску (вырезаем часть - делаем черным)
                # inv_mask делает черным там, где на маске 255
                masked_img = img.copy()
                masked_img[mask == 255] = 0

                # Пути сохранения
                save_name = f"{base_name}_v{i}.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, m_type, 'original', save_name), img)
                cv2.imwrite(os.path.join(OUTPUT_DIR, m_type, 'masks', save_name), mask)
                cv2.imwrite(os.path.join(OUTPUT_DIR, m_type, 'masked', save_name), masked_img)

    print("Готово! Датасет сформирован в папке", OUTPUT_DIR)


if __name__ == "__main__":
    process_images()
