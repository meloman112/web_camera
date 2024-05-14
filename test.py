import time
import os
from funcs import compute_sim, check_face
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.metrics import dp
from kivy.uix.textinput import TextInput
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
import insightface
import cv2
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


TRESHOLD_ADD_DB = 0.6

# Глобальные настройки
Window.size = (800, 600)
Window.clearcolor = (255/255, 186/255, 3/255, 1)
Window.title = "Добавить в базу"

class FaceRecognitionApp(App):

    def build(self):
        layout = BoxLayout(orientation='vertical')
        self.camera = Camera(play=True)
        layout.add_widget(self.camera)

        # Создаем поля ввода для имени, фамилии и ID с кастомными свойствами
        self.textinput_name = TextInput(hint_text='Имя', multiline=False,
                                        size_hint=(None, None), size=(dp(200), dp(40)),
                                        background_color=(1, 1, 1, 0.5),  # Цвет фона поля (полупрозрачный белый)
                                        foreground_color=(0, 0, 0, 1),  # Цвет текста (черный)
                                        font_size=dp(16))  # Размер шрифта
        self.textinput_surname = TextInput(hint_text='Фамилия', multiline=False,
                                           size_hint=(None, None), size=(dp(200), dp(40)),
                                           background_color=(1, 1, 1, 0.5),  # Цвет фона поля (полупрозрачный белый)
                                           foreground_color=(0, 0, 0, 1),  # Цвет текста (черный)
                                           font_size=dp(16))  # Размер шрифта
        self.textinput_id = TextInput(hint_text='ID', multiline=False,
                                      size_hint=(None, None), size=(dp(200), dp(40)),
                                      background_color=(1, 1, 1, 0.5),  # Цвет фона поля (полупрозрачный белый)
                                      foreground_color=(0, 0, 0, 1),  # Цвет текста (черный)
                                      font_size=dp(16))  # Размер шрифта

        # Создаем кнопку для добавления в базу
        btn_add_to_db = Button(text='Добавить в базу', size_hint=(None, None), size=(dp(150), dp(50)),
                               background_color=(0, 0.7, 0.3, 1),  # Цвет фона кнопки (зеленый)
                               color=(1, 1, 1, 1))
        btn_add_to_db.bind(on_press=self.add_frame_to_db)  # Привязываем метод к событию нажатия кнопки

        # Добавляем все в компоновку
        layout.add_widget(self.textinput_name)
        layout.add_widget(self.textinput_surname)
        layout.add_widget(self.textinput_id)
        layout.add_widget(btn_add_to_db)

        # Остальной код остается без изменений
        self.face_detection = insightface.app.FaceAnalysis(allowed_modules='detection')
        self.face_recognition = insightface.app.FaceAnalysis()
        self.face_detection.prepare(ctx_id=-1, det_size=(640, 480))
        self.face_recognition.prepare(ctx_id=-1, det_size=(640, 480))
        self.count = 0
        self.mongodb = MongoClient(os.getenv('MONGODB_LOCAL')).face_detection['clients']
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Обновление с частотой 30 кадров в секунду
        return layout

    def update(self, dt):
        frame = self.camera.texture
        if frame is not None:
            frame = np.frombuffer(frame.pixels, dtype=np.uint8)
            frame = frame.reshape(self.camera.texture.height, self.camera.texture.width, 4)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            self.count += 1
            faces = self.face_detection.get(frame)
            self.current_frame = frame  # Сохраняем текущий кадр для последующего использования

            # Преобразуем кадр в текстуру и отображаем его
            frame = cv2.flip(frame, 1)  # Отразить кадр по вертикали (по умолчанию он отображается вверх ногами)
            buf1 = cv2.flip(frame, 0).tobytes()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf1, colorfmt='bgr', bufferfmt='ubyte')
            self.camera.texture = texture1

    def add_frame_to_db(self, instance):
        if hasattr(self, 'current_frame'):  # Проверяем, был ли сохранен текущий кадр
            # Получаем значения из полей ввода
            name = self.textinput_name.text
            surname = self.textinput_surname.text
            person_id = self.textinput_id.text

            # Передаем текущий кадр и данные в функцию add_to_db
            self.add_to_db(self.current_frame, name, surname, person_id)

    def add_to_db(self, img, name, surname, person_id):
        face = self.face_recognition.get(img)
        face_data = face[0]
        embedding = face_data.embedding.tolist()

        processed_item = {
            'person_id': person_id,
            'name': name,
            'surname': surname,
            'embedding': embedding,
            'det_score': round((float(face_data.det_score) * 100), 3),
            'angle': face_data.pose.tolist(),
            'landmark_3d': face_data.landmark_3d_68.tolist(),
            "update_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            'utilized': 0,
        }
        self.mongodb.insert_one(processed_item)
        print('add to db')


if __name__ == '__main__':
    FaceRecognitionApp().run()

