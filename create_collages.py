import cv2
import os
import numpy as np

def create_collage(image_paths, collage_size=(2, 2)):
    images = [cv2.imread(path) for path in image_paths]
    # Предполагаем, что все изображения одинакового размера
    image_height, image_width = images[0].shape[:2]

    collage_image = np.zeros((image_height * collage_size[0], image_width * collage_size[1], 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        y = idx // collage_size[1] * image_height
        x = idx % collage_size[1] * image_width
        collage_image[y:y+image_height, x:x+image_width] = img

    return collage_image

def process_images(folder_path):
    # Получение и сортировка имен файлов
    filenames = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Группировка файлов по 16 изображений
    for i in range(0, len(filenames), 16):
        group = filenames[i:i + 16]
        if len(group) < 16:
            break

        collage = create_collage([os.path.join(folder_path, f) for f in group], collage_size=(4, 4))
        cv2.imwrite(os.path.join(folder_path, f'collages/{i//16}.jpg'), collage)

# Путь к папке с изображениями
process_images("/images")
