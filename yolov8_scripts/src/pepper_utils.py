import collections
import random
from typing import List, Dict

import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
from shapely import Polygon
import geopandas as gpd
from yolov8_scripts.src.pepper_fruit import PepperFruit
from yolov8_scripts.src.pepper_peduncle import PepperPeduncle
# from yolov8_scripts.src.detected_frame import OneFrame
import math
# from figures import BLUE, GRAY, RED, set_limits
from shapely.plotting import plot_polygon, plot_points

def get_img_size(img_path):
    img = read_image(img_path)
    return img.shape
def get_all_image_path_in_folder(path):
    img_list = list()
    for dirs, subdir, files in os.walk(path):
        for file_name in files:
            if file_name.endswith(".jpeg") or file_name.endswith(".jpg") or file_name.endswith(".png"):
                rgb_file = dirs + os.sep + file_name
                img_list.append(rgb_file)
    # print("all images in folder: ", img_list)
    return img_list[:]
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
def draw_bounding_box(confidence, x, y, w, h, color, fill=False):
    # Get the current reference
    ax = plt.gca()

    # plot the bounding box
    rect = patches.Rectangle((x-w/2, y-h/2), w, h, linewidth=2, edgecolor=color, facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    # plot the centroid of pepper
    # plt.plot(x, y, 'b*')

    # plot conf
    plt.text(x - w / 2, y - h / 2, round(float(confidence[0]), 2), color='red', fontsize=10) #round(confidence[0], 2)
def draw_bounding_polygon(confidence,mask, img_shape, color, fill=True):
    """
    Draw the bounding polygons associated with a peduncle.
    :param confidence: Confidence score associated with a mask.
    :param mask: Mask associated with a peduncle with values between 0 and 1.
    :param img_shape: The shape of the image with
    :return: None.
    """
    # print("mask: ", len(mask[0]), len(mask[0][0]))
    # print("mask: ", mask)

    mask = torch.transpose(mask, 0, 1)
    polygon = Polygon(mask.T)
    x, y = polygon.exterior.xy
    # print('img_shape: ', img_shape)
    x_scaled = [i * img_shape[1] for i in x]
    y_scaled = [i * img_shape[0] for i in y]
    # print('x, y: ', x, y)
    if fill:
        plt.fill(x_scaled, y_scaled, color=color, alpha=0.7)
    else:
        plt.fill(x_scaled, y_scaled, color=color, alpha=1)
    plt.plot(*polygon.exterior.xy)
    # p = gpd.GeoSeries(polygon)
    # p.plot()


def put_title(detected_frame):
    # displaying the title
    plt.title(label=f"Pepper: {len(detected_frame.pepper_fruit_detections)} Peduncle: {len(detected_frame.pepper_peduncle_detections)}",
              fontsize=10,
              color="black")
def draw_pepper(one_frame):
    img = np.asarray(Image.open(one_frame.img_path))
    img_name = one_frame.img_path.split('/')[-1].split('.')[0]
    plt.imshow(img)

    put_title(one_frame)
    for peduncle in one_frame.pepper_peduncle_detections.values():
        mask = peduncle.mask
        draw_bounding_polygon(peduncle.conf, mask, one_frame.img_shape, color='black', fill=False)
        # plt.plot(peduncle.xywh[0], peduncle.xywh[1], '*')
        # plt.text(peduncle.xywh[0], peduncle.xywh[1], f'({peduncle.xywh[0], peduncle.xywh[1]})')
    for pepper_fruit in one_frame.pepper_fruit_detections.values():
        xywh = pepper_fruit.xywh
        # plt.plot(pepper_fruit.xywh[0], pepper_fruit.xywh[1], '*')
        # plt.text(pepper_fruit.xywh[0], pepper_fruit.xywh[1], f'({pepper_fruit.xywh[0], pepper_fruit.xywh[1]})', color="blue")
        x = int(xywh[0])
        y = int(xywh[1])
        w = int(xywh[2])
        h = int(xywh[3])
        draw_bounding_box(pepper_fruit.conf, x, y, w, h, color="black",fill=False)

    for idx, pepper in one_frame.pepper_detections.items():
        r = np.round(np.random.rand(), 1)
        g = np.round(np.random.rand(), 1)
        b = np.round(np.random.rand(), 1)
        # a = np.round(np.clip(np.random.rand(), 0, 1), 1)
        color = (r, g, b)
        pepper_fruit = pepper.pepper_fruit
        pepper_peduncle = pepper.pepper_peduncle
        xywh = pepper_fruit.xywh
        x = int(xywh[0])
        y = int(xywh[1])
        w = int(xywh[2])
        h = int(xywh[3])
        draw_bounding_box(pepper_fruit.conf, x, y, w, h, color=color)

        mask = pepper_peduncle.mask
        draw_bounding_polygon(pepper_peduncle.conf, mask, one_frame.img_shape, color=color)
    # plt.axis('off')
    plt.savefig(f"/home/jy/PycharmProjects/Perception-Resources/yolov8_scripts/src/results_7/{img_name}_pepper_result.png",
                bbox_inches='tight', pad_inches = 1)
    # / home / jy / PycharmProjects / Perception - Resources / yolov8_scripts / src / results_3 / IMG_0971__1_fruit_result.png
    plt.clf()
    plt.cla()

def draw_pepper_fruits(detected_frame):
    img = np.asarray(Image.open(detected_frame.img_path))
    img_name = detected_frame.img_path.split('/')[-1].split('.')[0]
    plt.imshow(img)

    put_title(detected_frame)
    for pepper in detected_frame.pepper_fruit_detections:
        print('drawing one pepper')

        xywh = pepper.xywh
        x = int(xywh[0])
        y = int(xywh[1])
        w = int(xywh[2])
        h = int(xywh[3])
        draw_bounding_box( pepper.conf, x, y, w, h)

    plt.savefig(f"/home/jy/PycharmProjects/Perception-Resources/yolov8_scripts/src/results_7/{img_name}_fruit_result.png")
    plt.clf()
    plt.cla()
def draw_pepper_peduncles(detected_frame):
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
    output += f"folder_path={pd._path}\n# of detected_images: {len(pd.detected_frames)}\n"
    for detected_img in pd.detected_frames:
        output += f"\t{detected_img.img_path.split('/')[-1]}\n"
        output += f"\t\tDetected {detected_img.pepper_fruit_count} peppers\n"
        for pepper in detected_img.pepper_fruit_detections:
            output += f"\t\t\t{pepper}\n"
        output += f"\t\tDetected {detected_img.pepper_peduncle_count} peduncles\n"
        # for peduncle in detected_img.pepper_peduncle_detections:
            # output += f"\t\t\t{peduncle}\n"
    output += "============================================\n"
    return output

def get_image_from_webcam():
    camera = cv2.VideoCapture(0)
    while True:
        return_value,image = camera.read()
        # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = cv2.flip(image, 1) # <class 'numpy.ndarray'>
        cv2.imshow('image',image)

        if cv2.waitKey(1)& 0xFF == ord('s'):
            cv2.imwrite('test.jpg',image)
            break
    camera.release()
    cv2.destroyAllWindows()
    return image

# def match_pepper_fruit_peduncle(detected_frame: OneFrame):
#     pepper_fruit_detections = detected_frame.pepper_fruit_detections
#     pepper_peduncle_detections = detected_frame.pepper_peduncle_detections
#
#     pepper_fruit_peduncle_distances = []
#     for pepper_fruit in pepper_fruit_detections:
#         for pepper_peduncle in pepper_peduncle_detections:
#             pepper_fruit_peduncle_distances.append(distance_between_pepper_fruit_peduncle(pepper_fruit, pepper_peduncle))
#     pepper_fruit_peduncle_match = remove_duplicate_peduncles(pepper_fruit_peduncle_distances)
#     return pepper_fruit_peduncle_match

def is_above(pepper_fruit:PepperFruit, pepper_peduncle:PepperPeduncle):
    # print(pepper_fruit.xywh[1], pepper_peduncle.xywh[1])
    return pepper_fruit.xywh[1]>=pepper_peduncle.xywh[1]
def match_pepper_fruit_peduncle(pepper_fruit_detections: Dict[int, PepperFruit],pepper_peduncle_detections: Dict[int, PepperPeduncle]):

    pepper_fruit_peduncle_distances = []
    for pepper_fruit in pepper_fruit_detections.values():
        min_dist = math.inf
        peduncle_match = None
        for pepper_peduncle in pepper_peduncle_detections.values():
            dist = distance_between_pepper_fruit_peduncle(pepper_fruit, pepper_peduncle)
            x, y, w, h = pepper_fruit.xywh
            box1 = [[x-w/2 , y-h/2], [x+w/2 , y-h/2], [x+w/2 , y+h/2], [x-w/2 , y+h/2]]
            x, y, w, h = pepper_peduncle.xywh
            box2 = [[x-w/2 , y-h/2], [x+w/2 , y-h/2], [x+w/2 , y+h/2], [x-w/2 , y+h/2]]
            iou = calculate_iou(box1, box2)
            # print("iou", iou)
            if dist<min_dist and iou>0:
                peduncle_match = pepper_peduncle
                min_dist = dist
        if not peduncle_match:
            peduncle_match = PepperPeduncle(-1)
        pepper_fruit_peduncle_distances.append(((pepper_fruit.number, peduncle_match.number), min_dist))

    pepper_fruit_peduncle_match = remove_duplicate_peduncles(pepper_fruit_peduncle_distances)
    return pepper_fruit_peduncle_match
def choose_unmatching(duplicate_list):
    # ((pf_number, pp_number), distance)
    pepper_delete = list()
    for duplicates in duplicate_list:
        duplicates = sorted(duplicates, key=lambda d: d[1])
        pepper_delete.append(duplicates[1:])
    return pepper_delete
def remove_duplicate_peduncles(pepper_fruit_peduncle_distances: list):
    # remove duplicate peduncles

    detetected_pepper_fruit = []
    detected_pepper_peduncle = []
    clean_pepper_fruit_peduncle_distances = []

    for (pf, pp), d in pepper_fruit_peduncle_distances:
        detetected_pepper_fruit.append(pf)
        detected_pepper_peduncle.append(pp)
    duplicate_pepper_fruit = [item for item, count in collections.Counter(detetected_pepper_fruit).items() if count > 1]
    duplicate_pepper_peduncle = [item for item, count in collections.Counter(detected_pepper_peduncle).items() if count > 1]

    pf_duplicate_list = list()
    for pepper_fruit in duplicate_pepper_fruit: # index of peppers
        duplicate_list = list()
        for i in range(len(pepper_fruit_peduncle_distances)):
            (pf, pp), d = pepper_fruit_peduncle_distances[i]
            if pf == pepper_fruit:
                duplicate_list.append(pepper_fruit_peduncle_distances[i])
        pf_duplicate_list.append(duplicate_list)
    pepper_fruit_delete = choose_unmatching(pf_duplicate_list)
    for d in pepper_fruit_peduncle_distances:
        if d in pepper_fruit_delete:
            pepper_fruit_peduncle_distances.remove(d)

    pp_duplicate_list = list()
    for pepper_peduncle in duplicate_pepper_peduncle: # index of peppers
        duplicate_list = list()
        for i in range(len(pepper_fruit_peduncle_distances)):
            (pf, pp), d = pepper_fruit_peduncle_distances[i]
            if pp == pepper_peduncle:
                duplicate_list.append(pepper_fruit_peduncle_distances[i])
        pp_duplicate_list.append(duplicate_list)
    pepper_peduncle_delete = choose_unmatching(pp_duplicate_list)
    for d in pepper_fruit_peduncle_distances:
        if d in pepper_peduncle_delete:
            pepper_fruit_peduncle_distances.remove(d)

    return pepper_fruit_peduncle_distances

def distance_between_pepper_fruit_peduncle(pepper_fruit: PepperFruit, pepper_peduncle: PepperPeduncle):
    pepper_fruit_xywh = pepper_fruit.xywh
    pepper_peduncle_xywh = pepper_peduncle.xywh

    pepper_fruit_number = pepper_fruit.number
    pepper_peduncle_number = pepper_peduncle.number
    pf_coords = [pepper_fruit_xywh[0].item(),pepper_fruit_xywh[1].item()]
    pp_coords = [pepper_peduncle_xywh[0].item(), pepper_peduncle_xywh[1].item()]
    distance = math.dist(pf_coords, pp_coords)
    # return ((pepper_fruit_number, pepper_peduncle_number),distance)
    return distance

def red_to_green(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_lo = np.array([50, 0, 0])
    red_hi = np.array([255, 255, 255])
    # Mask image to only select browns
    mask = cv2.inRange(hsv, red_lo, red_hi)
    cv2.imwrite("mask.jpg", mask)
    # img[mask > 0] = img[mask > 0] * [0.3, 0, 0] + [0, 130, 0]
    img[mask > 0] = 0
    return img

def red_to_green_2(img):
    b, g, r = cv2.split(img)  # get b,g,r
    rgb_img = cv2.merge([r, g, b])
    # plt.imshow(rgb_img)

    x, y, z = np.shape(img)
    red = np.zeros((x, y, z), dtype=int)
    green = np.zeros((x, y, z), dtype=int)
    blue = np.zeros((x, y, z), dtype=int)
    for i in range(0, x):
        for j in range(0, y):
            red[i][j][0] = rgb_img[i][j][0]
            green[i][j][1] = rgb_img[i][j][1]
            blue[i][j][2] = rgb_img[i][j][2]
    # plt.imshow(red)
    # # plt.show()
    # plt.imshow(green)
    # # plt.show()
    # plt.imshow(blue)
    # plt.show()

    retrack_original = np.zeros((x, y, z), dtype=int)
    for i in range(0, x):
        for j in range(0, y):
            retrack_original[i][j][0] = red[i][j][0]*0.2//1
            retrack_original[i][j][1] = green[i][j][1]
            retrack_original[i][j][2] = blue[i][j][2]
    # cv2.imwrite('ori.jpg', retrack_original)
    plt.imshow(retrack_original)
    plt.show()
    return retrack_original
def calculate_iou(box_1, box_2):
    '''

    :param box_1: 4 x 2 tl, tr, br, bl
    :param box_2:
    :return: iou from 0-1
    '''
    # fig = plt.figure(1, dpi=90)
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)

    # plot_polygon(poly_1, add_points=False, color='red')
    # plot_polygon(poly_2, add_points=False, color='blue')
    # plt.savefig(f"/home/jy/PycharmProjects/Perception-Resources/yolov8_scripts/src/results_5/{int(np.random.rand()*100)}.png")
    # plt.clf()
    # plt.cla()

    iou = poly_1.intersection(poly_2).area/poly_1.union(poly_2).area
    return iou

if __name__ == '__main__':
    # get_image_from_webcam()
    # image = cv2.imread("/home/jy/PycharmProjects/Perception-Resources/dataset/colorful/bell-peppers-do-the-different-colors-really-taste-any-different.jpg")
    image = cv2.imread("/home/jy/PycharmProjects/Perception-Resources/dataset/colorful/bell-peppers-in-season.jpg")
    # cv2.imwrite("img1.jpg", image)
    # image = red_to_green(image)
    # cv2.imwrite("img2.jpg", image)
    red_to_green_2(image)