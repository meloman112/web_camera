import os
import cv2
import numpy as np
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

SAMPLE_IMAGE_PATH = "./images/sample/"


def check_image(image):
    if image is None:
        print("image is not valid")
        return False
    return True


def test(image_name, model_dir="./resources/anti_spoof_models", device_id=0):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = image_name
    result = check_image(image)
    if result is False:
        return
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    print(label)
    if (label == 1 and value >= 0.8):
        print(f"Image is Real Face. Score: {value}. {image.shape}")
        return True
    else:
        print(f"Image is Fake Face. Score: {value}.")
        return False


if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        else:
            # print('image:', image)
            test(frame, "./resources/anti_spoof_models", 0)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break
        # Закрываем окно и освобождаем ресурсы
    cv2.destroyAllWindows()
    capture.release()
