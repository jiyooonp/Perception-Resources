import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import ultralytics
import numpy as np
import os

ultralytics.checks()
# Basic Parameters
model = YOLO("../weights/best.pt")
path = '/home/jy/PycharmProjects/Perception-Resources/dataset/testbed'

classes = ["pepper"]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


def main_yay(path):
    print("in main yay")
    imgs_path = all_images_in_folder(path)[:4]
    all_boxes_list, all_masks_list, all_probs_list = predict_peppers(imgs_path, show_result=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def all_images_in_folder(path):
    img_list = list()
    for dirs, subdir, files in os.walk(path):
        for file_name in files:
            if file_name.endswith(".jpeg"):
                rgb_file = dirs + os.sep + file_name
                img_list.append(rgb_file)
    print("all images in folder: ", img_list)
    return img_list
def predict_pepper(img_path, show_result: bool = True, print_result: bool = False):
    print("predicting one pepper")
    img = read_image(img_path)
    results = model(img)

    boxes_list = list()
    masks_list = list()
    probs_list = list()
    # print(results)

    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        # print("box",boxes)
        # masks = result.masks  # Masks object for segmenation masks outputs
        probs = result.probs  # Class probabilities for classification outputs
        # print("prob",probs)
        boxes_list.append(boxes)
        # masks_list.append(masks)
        probs_list.append(probs)

    # if show_result:
    #     for result in results:
    #         res_plotted = result[0].plot()
    #         cv2.imshow("result", res_plotted)
    if print_result:
        for result in results:
            print_result_boxes(result)
    return boxes_list, masks_list, probs_list
def read_image(img_path):
    img = cv2.imread(img_path)
    # print(img)
    # cv2.imshow("image", img)
    # print("showed image")
    # cv2.waitKey(0)
    # cv2.distroyAllWindows()
    return img
def predict_peppers(imgs_path: list, show_result: bool = False, print_result: bool = True):

    all_boxes_list = list()
    all_masks_list = list()
    all_probs_list = list()

    for img_path in imgs_path:
        boxes_list, masks_list, probs_list = predict_pepper(img_path, show_result, print_result)
        all_boxes_list.append(boxes_list)
        all_masks_list.append(masks_list)
        all_probs_list.append(probs_list)

    return all_boxes_list, all_masks_list, all_probs_list
def print_result_boxes(result):
    boxes = result.boxes  # Boxes object for bbox outputs
    print("box", boxes)
    for box in boxes:
        print(f"detected {[box.cls]}")
        x, y, w, h = box.xywh

        print("xywh:", x, y, w, h)
        # draw_bounding_box(result.orig_img, box.cls, box.conf, x, y, x + w, y - h)
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
    return results
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

if __name__ == '__main__':
    # main_yay(path)
    model = YOLO("../weights/best.pt")
    img_path = all_images_in_folder(path)[0]
    inputs = read_image(img_path)
    results = model(inputs)

    boxes = results[0].boxes
    box = boxes[0]  # returns one box
    print(box.xyxy)