import os
from funcs import compute_sim, check_face
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
import insightface
import cv2
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
from test1 import test
load_dotenv()


TRESHOLD_ADD_DB = 0.6

# Глобальные настройки
Window.size = (800, 600)
Window.clearcolor = (255/255, 186/255, 3/255, 1)
Window.title = "Распознавание лиц"

class FaceRecognitionApp(App):

    def build(self):
        layout = BoxLayout(orientation='vertical')

        # Заменяем камеру Kivy на камеру OpenCV
        self.cap = cv2.VideoCapture("http://192.168.0.106:4747/video")

        self.face_detection = insightface.app.FaceAnalysis(allowed_modules='detection')
        self.face_recognition = insightface.app.FaceAnalysis()
        self.face_detection.prepare(ctx_id=-1, det_size=(640, 480))
        self.face_recognition.prepare(ctx_id=-1, det_size=(640, 480))
        self.count = 0
        self.mongodb = MongoClient(os.getenv('MONGODB_LOCAL')).face_detection['clients']
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Обновление с частотой 30 кадров в секунду
        return layout

    def update(self, dt):
        ret, frame = self.cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = cv2.flip(frame, 1)
            buf1 = frame.tostring()

            texture1 = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='rgba')
            texture1.blit_buffer(buf1, colorfmt='rgba', bufferfmt='ubyte')
            self.camera.texture = texture1

            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            self.count += 1
            faces = self.face_detection.get(frame)
            if faces is not None:
                for face in faces:
                    if self.count % 30 == 0:
                        bbox = check_face(face)
                        if bbox is not False:
                            x1, y1, x2, y2 = [int(val) for val in bbox]
                            # Получение размеров изображения
                            height, width, _ = frame.shape
                            x1 = max(0, x1 - 20)
                            y1 = max(0, y1 - 20)
                            x2 = min(width, x2 + 20)
                            y2 = min(height, y2 + 20)
                            cropped_image = frame[y1:y2, x1:x2]
                            if test(cropped_image):
                                name = self.recognition_frame(cropped_image)
                                print(name)
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Преобразуем кадр в текстуру и отображаем его
            frame = cv2.flip(frame, 1)  # Отразить кадр по вертикали (по умолчанию он отображается вверх ногами)
            buf1 = cv2.flip(frame, 0).tobytes()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf1, colorfmt='bgr', bufferfmt='ubyte')
            self.camera.texture = texture1

    def recognition_frame(self, face_frame):
        face_data = self.face_recognition.get(face_frame)
        try:
            if np.all(face_data[0].embedding) == 0:
                print('False')
                return 0
            for value in self.mongodb.find():
                db_emb = np.array(value['embedding'])
                score = compute_sim(face_data[0].embedding, db_emb)
                print(score)
                if score is None:
                    return 0, 0
                elif score > TRESHOLD_ADD_DB:
                    return value['name']
        except Exception as e:
            print(e)
            return 0

if __name__ == '__main__':
    FaceRecognitionApp().run()
