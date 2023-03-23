import random

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO
import ultralytics
import numpy as np
import os
from shapely import Polygon
import geopandas as gpd


def get_all_image_path_in_folder(path):
    img_list = list()
    for dirs, subdir, files in os.walk(path):
        for file_name in files:
            if file_name.endswith(".jpeg") or file_name.endswith(".jpg"):
                rgb_file = dirs + os.sep + file_name
                img_list.append(rgb_file)
    print("all images in folder: ", img_list)
    return img_list
def read_image(img_path):
    img = cv2.imread(img_path)
    img = np.asarray(img)
    return img
def print_result_boxes(detected_img):
    print(f"detected {detected_img.pepper_fruit_count} peppers!")
    for pepper in detected_img.pepper_fruit_detections:
        print(pepper)
        # draw_bounding_box(result.orig_img, box.cls, box.conf, x, y, x + w, y - h)
def print_result_masks(detected_img):
    print(f"detected {detected_img.pepper_peduncle_count} peduncles!")
    for peduncle in detected_img.pepper_peduncle_detections:
        print(peduncle)
        # draw_bounding_box(result.orig_img, box.cls, box.conf, x, y, x + w, y - h)
def draw_bounding_box(confidence, x, y, w, h):
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
def draw_bounding_polygon(confidence,mask, img_shape):
    """
    Draw the bounding polygons associated with a peduncle.
    :param confidence: Confidence score associated with a mask.
    :param mask: Mask associated with a peduncle with values between 0 and 1.
    :param img_shape: The shape of the image with
    :return: None.
    """
    # print("mask: ", len(mask[0]), len(mask[0][0]))
    # print("mask: ", mask)

    polygon = Polygon(mask[0])
    x, y = polygon.exterior.xy
    print('img_shape: ', img_shape)
    x_scaled = [i * img_shape[1] for i in x]
    y_scaled = [i * img_shape[0] for i in y]
    print('x, y: ', x, y)

    plt.plot(x_scaled, y_scaled)
    # plt.plot(*polygon.exterior.xy)
    # p = gpd.GeoSeries(polygon)
    # p.plot()

    plt.plot(100, 100, 'b*')

def put_title(detected_frame):
    # displaying the title
    plt.title(label=f"Pepper: {detected_frame.pepper_fruit_count} Peduncle: {detected_frame.pepper_peduncle_count}",
              fontsize=10,
              color="black")
def draw_peppers(detected_frame):
    img = np.asarray(Image.open(detected_frame.img_path))
    img_name = detected_frame.img_path.split('/')[-1].split('.')[0]
    plt.imshow(img)

    put_title(detected_frame)
    for pepper in detected_frame.pepper_fruit_detections:

        xywh = pepper.xywh
        x = int(xywh[0])
        y = int(xywh[1])
        w = int(xywh[2])
        h = int(xywh[3])
        draw_bounding_box( pepper.conf, x, y, w, h)

    plt.savefig(f"results_2/{img_name}_fruit_result.png")
    plt.clf()
    plt.cla()
def draw_peduncles(detected_frame):
    img = np.asarray(Image.open(detected_frame.img_path))
    img_name = detected_frame.img_path.split('/')[-1].split('.')[0]
    plt.imshow(img)
    put_title(detected_frame)
    for peduncle in detected_frame.pepper_peduncle_detections:
        mask = peduncle.mask
        draw_bounding_polygon(peduncle.conf,mask, detected_frame.img_shape)

    plt.savefig(f"results_2/{img_name}_peduncle_result.png")
    plt.clf()
    plt.cla()
def print_pepperdetection(pd):
    output = "\n============================================\n"
    output += f"PepperDetection Results:\n"
    output += f"folder_path={pd.path}\n# of detected_images: {len(pd.detected_frames)}\n"
    for detected_img in pd.detected_frames:
        output += f"\t{detected_img.img_path.split('/')[-1]}\n"
        output += f"\t\tDetected {detected_img.pepper_fruit_count} peppers\n"
        for pepper in detected_img.pepper_fruit_detections:
            output += f"\t\t\t{pepper}\n"
        output += f"\t\tDetected {detected_img.pepper_peduncle_count} peduncles\n"
        for peduncle in detected_img.pepper_peduncle_detections:
            output += f"\t\t\t{peduncle}\n"
    output += "============================================\n"
    return output

