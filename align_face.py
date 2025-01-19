import cv2
import dlib
import numpy as np
import os

# Загрузите предварительно обученную модель для обнаружения лицевых особенностей
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_predictor = dlib.shape_predictor(predictor_path)
face_detector = dlib.get_frontal_face_detector()

def align_face(image_path, desired_left_eye=(0.20, 0.45), desired_face_width=2256, desired_face_height=None):
    # Загрузите изображение
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц
    faces = face_detector(gray)
    if len(faces) == 0:
        print("Не удалось найти лицо")
        return None

    face = faces[0]

    # Обнаружение ключевых точек лица
    landmarks = face_predictor(gray, face)

    # Координаты глаз
    left_eye_center = np.array([float(landmarks.part(36).x + landmarks.part(39).x) / 2,
                                float(landmarks.part(36).y + landmarks.part(39).y) / 2])
    right_eye_center = np.array([float(landmarks.part(42).x + landmarks.part(45).x) / 2,
                                 float(landmarks.part(42).y + landmarks.part(45).y) / 2])

    # Вычислите угол и масштаб
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    desired_dist = desired_left_eye[0] * desired_face_width
    current_dist = np.sqrt((dX ** 2) + (dY ** 2))
    scale = desired_dist / current_dist

    # Вычисление центра между глаз
    eyes_center = (left_eye_center + right_eye_center) / 2

    # Матрица аффинного преобразования
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # Обновление матрицы трансформации для смещения
    tX = desired_face_width * 0.5
    tY = desired_face_height * desired_left_eye[1] if desired_face_height is not None else desired_face_width * desired_left_eye[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    # Применение преобразования
    output = cv2.warpAffine(img, M, (desired_face_width, desired_face_height if desired_face_height is not None else desired_face_width), flags=cv2.INTER_CUBIC)

    return output


def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            print(f"Обработка изображения: {image_path}")
            aligned_image = align_face(image_path)
            if aligned_image is not None:
                cv2.imwrite(os.path.join(folder_path, "aligned/" + filename), aligned_image)

# Замените "./images" на "/images", если папка images находится в корне файловой системы контейнера
process_folder("/images")

# # Пример использования
# aligned_image = align_face("./images/your_image.jpg")
# if aligned_image is not None:
#     cv2.imwrite("./images/aligned_your_image.jpg", aligned_image)
