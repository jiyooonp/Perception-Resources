import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from skimage.measure import label
from scipy.optimize import curve_fit
from skimage.morphology import closing, medial_axis

from yolov8_scripts.src.one_frame import OneFrame
from yolov8_scripts.src.pepper_fruit_detector import PepperFruitDetector
from yolov8_scripts.src.pepper_peduncle_detector import PepperPeduncleDetector
from yolov8_scripts.src.pepper_utils import *
from yolov8_scripts.src.pepper_peduncle_utils import *
# input: image
class Perception:
    def __init__(self, source, fps, threshold=0.5, save=True):
        self.source = source
        self.start_time = time.time()
        self.fps = fps
        self.save = save
        self.threshold = threshold
        self.pepper_fruits = dict()
        self.pepper_peduncles = dict()
        self.pepper = dict()
    def get_image(self):
        #################################################################
        # get image from source
        # output: RGBD information
        #################################################################
        if self.source == "webcam":
            self.image = get_image_from_webcam()
        else:
            self.image = read_image(self.source)
    def get_depth(self, image, x, y):
        #################################################################
        # given an image and x, y coordinates, return the depth information
        # note this will take in an area in the image, average the depth
        # and do additional calculation to compensate for the noise given
        # by the RGBD camera
        # input:
        #   image: (H, W, D)?
        #   x, y: coordinates
        # output:
        #   d: depth of that point
        #################################################################
        pass
    def process_image(self):
        #################################################################
        # DO_NOT_DO
        # process the image to match the dim for yolo (prob don't need this)
        #################################################################
        pass

    #####################################################################
    # When base goes to one location, use the long range images and retreive
    # all the locations of peppers
    #####################################################################
    def detect_peppers_one_frame(self, path, thresh=0.5):
        #################################################################
        # use yolov8_scripts and get the pepper locations
        # input:
        #   path: image path
        #   thresh: disgard detection lower than threshold
        # output:
        #   locations: all the locations of the pepper boxes [conf, x, y] (N, 3)
        #################################################################
        one_frame = OneFrame(path)
        one_frame.run()
        self.pepper = one_frame.pepper_detections
    def detect_peppers_in_folder(self):
        files = get_all_image_path_in_folder(self.source)
        for path in files:
            self.detect_peppers_one_frame(path)
    def detect_peppers_time_frame(self, frames, thresh=0.5):
        #################################################################
        # JIYOON TODO
        # stack pepper locations over a timeframe time
        # input:
        #   frames: F number of frames to be stored in list
        # output:
        #   locations: all the locations of the pepper boxes over a
        #       number of frames F x [conf, x, y] (F, N, 3)
        #################################################################
        for i in range(frames):
            pepper_fruit_detection = self.detect_peppers_one_frame(i, thresh)
            self.pepper_fruit[i] = pepper_fruit_detection # store dictionary of pepper_fruit in frame number
        # print(self.pepper_fruit)
    def clear_false_positives(self):
        #################################################################
        # TODO
        # algorithm to take in a series of bounding boxes and output
        # true positive pepper locations
        # input:
        #   locations : pepper locations over F frames (F, N, 3)
        # output:
        #   self.pepper_locations: true positive pepper locations including
        #   the depth information (M, 4)
        #################################################################
        pass

    #####################################################################
    # Once the manipulator goes closer to the pepper, we have one pepper
    # as target.
    #####################################################################
    def set_target_pepper(self, pepper_index):
        #################################################################
        # Using the pepper_index, take another closer image of the pepper,
        # run the detection algorithm to get a more precise bounding box.
        # Store the pepper's information in self.pepper
        # output:
        #   self.pepper = {"idx": None, "box": (L, T, R, D), "location": (xc, yc, d)}
        #################################################################
        pass
    def get_peduncle_location(self):
        #################################################################
        # for self.pepper, crop the image, run the segmentation model and
        # get the segmented peduncle mask.
        # output:
        #   self.peduncle_mask: idk what this form is
        #################################################################
        self.peduncle_masks = self.pepper.pepper_peduncle_detections
    def get_point_of_interaction(self):
        #################################################################
        # using self.peduncle_mask, calculate the point of interaction
        # input:
        #   self.peduncle_mask
        # output:
        #   self.poi: (x, y, d)
        #################################################################

        for key, single_pepper in self.pepper.items():
            mask = single_pepper.pepper_peduncle.mask
            pepper_fruit_xywh = single_pepper.pepper_fruit.xywh
            pepper_peduncle_xywh = single_pepper.pepper_peduncle.xywh

            single_pepper.pepper_peduncle.curve = fit_curve_to_mask(mask, pepper_fruit_xywh, pepper_peduncle_xywh)

            total_curve_length = single_pepper.pepper_peduncle.curve.full_curve_length()

            poi_x, poi_y = determine_poi(single_pepper.pepper_peduncle.curve, percentage, total_curve_length)
            single_pepper.pepper_peduncle.poi = (poi_x, poi_y)

    def get_peduncle_orientation(self):
        #################################################################
        # ISHU TODO        # calculate the orientation of the peduncle using self.peduncle_mask
        # output:
        #   self.peduncle_orienation: (x,y,z)
        #################################################################
        for key, single_pepper in self.pepper.items():
            curve = single_pepper.pepper_peduncle.curve
            poi = single_pepper.pepper_peduncle.poi
            pepper_fruit_xywh = single_pepper.pepper_fruit.xywh
            pepper_peduncle_xywh = single_pepper.pepper_peduncle.xywh

            point_x, point_y = determine_next_point(curve, poi, pepper_fruit_xywh, pepper_peduncle_xywh)

            poi_z = self.get_depth(img, poi[0], poi[1])
            point_z = self.get_depth(img, point_x, point_y)

            return point_x - poi[0], point_y - poi[1], point_z - poi_z

    #####################################################################
    # ROS related
    #####################################################################


    def send_to_manipulator(self):
        #################################################################
        # send the point of interaction to the manipulator over ROS
        #################################################################
        pass


    #####################################################################
    # VISUALIZATION related
    #####################################################################

    
    def send_to_gui(self):
        #################################################################
        # send information to gui over ros
        #################################################################
        pass
    def get_from_gui(self):
        #################################################################
        # get information from gui over ros
        # such as commands (stop running/change fps/etc)
        #################################################################
        pass

