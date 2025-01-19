FROM python:3.9-slim

# Установите необходимые библиотеки
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    git \
    cmake \
    g++ \
    wget

# Установите Python библиотеки
RUN pip install numpy opencv-python dlib

RUN apt-get update && apt-get install -y libgl1

# Копируйте скрипт в контейнер
COPY align_face.py /align_face.py
COPY create_collages.py /create_collages.py
COPY shape_predictor_68_face_landmarks.dat /shape_predictor_68_face_landmarks.dat


# Запустите скрипт при старте контейнера
CMD ["python", "/align_face.py"]
