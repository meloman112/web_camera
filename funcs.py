import numpy as np


def compute_sim(feat1, feat2,):
    from numpy.linalg import norm

    try:
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim
    except Exception as e:
        print(e)
        return None

def check_face(face_data):
    if calculate_rectangle_area(face_data.bbox) > 1700:
        return face_data.bbox
    elif face_data.det_score > 0.7 and calculate_rectangle_area(face_data.bbox):
        return face_data.bbox
    else:
        return False

def calculate_rectangle_area(bbox):
    # Проверяем, что список bbox содержит четыре элемента
    if len(bbox) != 4:
        raise ValueError(
            "bbox должен содержать четыре координаты: x_min, y_min, x_max, y_max"
        )

    # Вычисляем ширину и высоту прямоугольника
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    # Вычисляем площадь прямоугольника
    area = width * height

    return area
