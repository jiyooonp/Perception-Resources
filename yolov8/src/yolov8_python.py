import cv2
from ultralytics import YOLO
import ultralytics
import numpy as np
import os

ultralytics.checks()
# Basic Parameters
model = YOLO("weights/best.pt")
path = '/home/jy/PycharmProjects/Perception-Resources/dataset/testbed/'

classes = ["pepper"]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def predict_folder(path,file_ext,args):
    os.chdir(path)
    for file in os.listdir():
        if file.endswith("."+file_ext):
            file_path = f"{path}/{file}"
            results = model.predict(source=file_path, show=True, save=True, conf=0.4)

def webcam_prediction():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Realtime Image')
    img_counter = 0
    while True:
        ret, image = cam.read()
        if not ret:
            print('failed to grab frame')
            break
        k = cv2.waitKey(1)
        if k % 256 == 27:
            print('escape hit, closing the app')
            break
        # screenshots will be taken
        elif k % 256 == 32:
            # the format for storing the images scrreenshotted
            img_name = f'opencv_frame_{img_counter}.jpg'
            # saves the image as a png file
            cv2.imwrite(img_name, image)
            print('screenshot taken')
            # the number of images automaticallly increases by 1
            img_counter += 1

        results = model.predict(source=image, save=True, save_txt=True)

        if results[0].probs:
            print(f"detected {len(results)} objects!")
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                image = result.orig_img
                for box in boxes:
                    print(f"detected {[box.cls]}")
                    x, y, w, h = box.xywh
                    draw_bounding_box(image, box.cls, box.conf, x, y, x + w, y - h)
        else:
            print(f"detected nothing")
            cv2.putText(image, "nothing found", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Input Image', image)
    cam.release()
    cam.destoryAllWindows()