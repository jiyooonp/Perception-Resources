import random

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO
import ultralytics
import numpy as np
import os

class PepperUtils:
    def __init__(self):
        self.classes = ["pepper"]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
    def get_all_image_path_in_folder(self, path):
        img_list = list()
        for dirs, subdir, files in os.walk(path):
            for file_name in files:
                if file_name.endswith(".jpeg"):
                    rgb_file = dirs + os.sep + file_name
                    img_list.append(rgb_file)
        print("all images in folder: ", img_list)
        return img_list
    def read_image(self, img_path):
        img = cv2.imread(img_path)
        return img
    def print_result_boxes(self, detected_img):
        print(f"detected {detected_img.get_pepper_count()} peppers!")
        for pepper in detected_img.get_detections():
            print(pepper)
            # draw_bounding_box(result.orig_img, box.cls, box.conf, x, y, x + w, y - h)
    def draw_bounding_box(self,confidence, x, y, w, h):
        # Get the current reference
        ax = plt.gca()

        # plot the bounding box
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, linewidth=2, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        # plot the centroid of pepper
        plt.plot(x, y, 'b*')

        # plot conf
        plt.text(x - w / 2, y - h / 2, confidence, color='red', fontsize=10)
    def put_title(self, detected_frame):
        # displaying the title
        plt.title(label=f"{detected_frame.get_pepper_count()} peppers detected",
                  fontsize=10,
                  color="black")
    def draw_img(self, detected_frame):
        img = np.asarray(Image.open(detected_frame.get_img_path()))
        img_name = detected_frame.get_img_path().split('/')[-1].split('.')[0]
        plt.imshow(img)

        self.put_title(detected_frame)
        for pepper in detected_frame.get_detections():

            xywh = pepper.get_xywh()
            x = int(xywh[0])
            y = int(xywh[1])
            w = int(xywh[2])
            h = int(xywh[3])
            self.draw_bounding_box( pepper.get_conf(), x, y, w, h)

        plt.savefig(f"results_2/{img_name}_result.png")
        plt.clf()
        plt.cla()
    def print_pepperdetection(self, pd):
        output = "\n============================================\n"
        output += f"PepperDetection Results:\n"
        output += f"folder_path={pd.path}\n# of detected_images: {len(pd.detected_frames)}\n"
        for detected_img in pd.detected_frames:
            output += f"\t{detected_img.get_img_path().split('/')[-1]}\n"
            output += f"\t\tDetected {detected_img.get_pepper_count()} peppers\n"
            for pepper in detected_img.get_detections():
                output += f"\t\t\t{pepper}\n"
        output += "============================================\n"
        return output