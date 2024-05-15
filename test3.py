import os
import cv2
import numpy as np
import insightface
from pymongo import MongoClient
from dotenv import load_dotenv
from test1 import test
load_dotenv()

TRESHOLD_ADD_DB = 0.6

# Установка параметров окна OpenCV
WINDOW_NAME = "Распознавание лиц"
cv2.namedWindow(WINDOW_NAME)

# Открываем видеопоток с камеры по указанному IP-адресу и порту
camera_ip = "http://192.168.0.106:4747/video"
cap = cv2.VideoCapture(camera_ip)

# Инициализируем модели для обнаружения и распознавания лиц
face_detection = insightface.app.FaceAnalysis(allowed_modules='detection')
face_detection.prepare(ctx_id=-1, det_size=(640, 480))
face_recognition = insightface.app.FaceAnalysis()
face_recognition.prepare(ctx_id=-1, det_size=(640, 480))

# Подключаемся к MongoDB
mongodb = MongoClient(os.getenv('MONGODB_LOCAL')).face_detection['clients']

# Основной цикл обработки кадров
while True:
    # Получаем кадр из видеопотока
    ret, frame = cap.read()

    if ret:
        # Преобразуем кадр в RGB, так как OpenCV читает в формате BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Обнаруживаем лица на кадре
        faces = face_detection.get(rgb_frame)

        # Если лица обнаружены, обрабатываем каждое лицо
        if faces is not None:
            for face in faces:
                # Обрезаем область лица
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                cropped_face = frame[y1:y2, x1:x2]

                # Проверяем лицо на соответствие базе данных
                if test(cropped_face):
                    # Получаем имя лица
                    face_data = face_recognition.get(cropped_face)
                    if np.all(face_data[0].embedding) != 0:
                        for value in mongodb.find():
                            db_emb = np.array(value['embedding'])
                            score = compute_sim(face_data[0].embedding, db_emb)
                            print(score)
                            if score is not None and score > TRESHOLD_ADD_DB:
                                name = value['name']
                                print(name)
                                break

                # Отрисовываем прямоугольник вокруг лица
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Отображаем кадр
        cv2.imshow(WINDOW_NAME, frame)

        # Обработка нажатия клавиши 'q' для выхода из цикла
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
