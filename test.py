import os
import time

import numpy as np
import cv2
from insightface.app import FaceAnalysis
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
app = FaceAnalysis()
app.prepare(ctx_id=0)

client = MongoClient(os.getenv('MONGODB_LOCAL'))

db = client.face_detection['clients']

img = cv2.imread('/home/meloman/Изображения/17128329716617c1cb474b3.jpg')
face = app.get(img)
face_data = face[0]
embedding = face_data.embedding.tolist()

processed_item = {
    'person_id': 1,
    'name': 'Babur',
    'embedding': embedding,
    'det_score': round((float(face_data.det_score) * 100), 3),
    'angle': face_data.pose.tolist(),
    'landmark_3d': face_data.landmark_3d_68.tolist(),
    "update_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    'utilized': 0,
}
db.insert_one(processed_item)