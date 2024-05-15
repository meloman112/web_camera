from src.anti_spoof_predict import AntiSpoofPredict
import os
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')


def check_image(image):
    if image is None:
        print("image is not valid")
        return False
    return True


def test(image_name, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image = image_name
    result = check_image(image)
    if result is False:
        return
    prediction = np.zeros((1, 3))
    test_speed = 0
    for model_name in os.listdir(model_dir):
        img = image
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        print(f"Image is Real Face. Score: {value}.")
        return True
    else:
        print(f"Image is Fake Face. Score: {value}.")
        return False
