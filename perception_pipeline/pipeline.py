import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from skimage.measure import label
from scipy.optimize import curve_fit
from skimage.morphology import closing, medial_axis

from yolov8_scripts.src.pepper_fruit_detector import PepperFruitDetector
from yolov8_scripts.src.pepper_peduncle_detector import PepperPeduncleDetector
from yolov8_scripts.src.pepper_utils import *
# input: image
class Perception:
    def __init__(self, source, fps, threshold=0.5, save=True):
        self.source = source
        self.start_time = time.now()
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
        # process the image to match the dim for yolo (prob don't need this)
        #################################################################
        pass

    #####################################################################
    # When base goes to one location, use the long range images and retreive
    # all the locations of peppers
    #####################################################################
    def detect_peppers_one_frame(self, thresh):
        #################################################################
        # use yolov8_scripts and get the pepper locations
        # input:
        #   thresh: disgard detection lower than threshold
        # output:
        #   locations: all the locations of the pepper boxes [conf, x, y] (N, 3)
        #################################################################
        PepperDetection = PepperFruitDetector(
            file_path='/home/jy/PycharmProjects/Perception-Resources/dataset/testbed_video_to_img',
            yolo_weight_path="/home/jy/PycharmProjects/Perception-Resources/yolov8_scripts/weights/pepper_fruit_best.pt")
        PepperDetection.run_detection(show_result=False)
        return PepperDetection.detected_frames[0].pepper_fruit_detections

    def detect_peppers_time_frame(self, thresh, frames):
        #################################################################
        # stack pepper locations over a timeframe time
        # input:
        #   frames: F number of frames to be stored in list
        # output:
        #   locations: all the locations of the pepper boxes over a
        #       number of frames F x [conf, x, y] (F, N, 3)
        #################################################################
        for i in range(frames):
            pepper_fruit_detection = self.detect_peppers_one_frame(thresh)
            self.pepper_fruit.append(pepper_fruit_detection)
        print(self.pepper_fruit)
    def detect_peduncles_one_frame(self, thresh):
        #################################################################
        # use yolov8_scripts and get the peduncle locations
        # input:
        #   thresh: disgard detection lower than threshold
        # output:
        #   locations: all the locations of the peduncle segments [conf, x, y] (N, 3)
        #################################################################
        PepperPeduncleDetection = PepperPeduncleDetector(
            file_path='/home/jy/PycharmProjects/Perception-Resources/dataset/peduncle',
            yolo_weight_path="../weights/pepper_peduncle_best.pt")
        PepperPeduncleDetection.run_detection()
        print(PepperPeduncleDetection)
        PepperPeduncleDetection.plot_results()
    def detect_peduncles_time_frame(self, thresh, frames):
        #################################################################
        # stack peduncle locations over a timeframe time
        # input:
        #   frames: F number of frames to be stored in list
        # output:
        #   locations: all the locations of the peduncle segments over a
        #       number of frames F x [conf, x, y] (F, N, 3)
        #################################################################
        for i in range(frames):
            pepper_peduncle_detection = self.detect_peduncles_one_frame(thresh)
            self.pepper_peduncles.append(pepper_peduncle_detection)
        print(self.pepper_peduncles)

    def get_pepper(self):
        pass
    def clear_false_positives(self):
        #################################################################
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
        self.peduncle_mask = None
    def get_point_of_interaction(self):
        #################################################################
        # using self.peduncle_mask, calculate the point of interaction
        # input:
        #   self.peduncle_mask
        # output:
        #   self.poi: (x, y, d)
        #################################################################
        closed_img = closing(self.peduncle_mask)
        medial_img, dist = medial_axis(closed_img, return_distance=True)
        labels, num = label(medial_img, return_num=True)

        poi_x = []
        poi_y = []

        for i in range(num):
            x, y = np.where(labels == i + 1)

            params1, cov1 = curve_fit(parabola, y, x)
            curve_x = parabola(y, params1[0], params1[1], params1[2])
            params2, cov2 = curve_fit(parabola, x, y)
            curve_y = parabola(x, params2[0], params2[1], params2[2])

            if np.linalg.norm(x - curve_x) < np.linalg.norm(y - curve_y):
                # Sorted assuming that the pepper is hanging to the left
                sy_x = np.array([x for _, x in sorted(zip(y, x))])
                sy_y = np.array([y for y, _ in sorted(zip(y, x))])

                a, b, c = params1
                curve_x = parabola(sy_y, a, b, c)
                full_length, _ = quad(dist_derivative, sy_y[0], sy_y[-1], args=(a, b))

                for j in range(len(sy_y)):
                    result, err = quad(dist_derivative, sy_y[0], sy_y[j], args=(a, b))
                    if abs(abs(result) - self.threshold * abs(full_length)) < 2:
                        poi_x.append(sy_y[j])
                        poi_y.append(curve_x[j])  # May have to choose point on medial_axis instead
                        break
            else:
                # Sorted assuming that the pepper is hanging upwards
                sx_x = np.array([x for x, _ in sorted(zip(x, y))])
                sx_y = np.array([y for _, y in sorted(zip(x, y))])

                a, b, c = params2
                curve_y = parabola(sx_x, a, b, c)
                full_length, _ = quad(dist_derivative, sx_x[0], sx_x[-1], args=(a, b))

                for j in range(len(sx_x)):
                    result, err = quad(dist_derivative, sx_x[0], sx_x[j], args=(a, b))
                    if abs(abs(result) - self.threshold * abs(full_length)) < 2:
                        poi_x.append(curve_y[j])  # May have to choose point on medial_axis instead
                        poi_y.append(sx_x[j])
                        break

        self.poi = np.array([poi_x, poi_y]).T

    def get_peduncle_orientation(self):
        #################################################################
        # calculate the orientation of the peduncle using self.peduncle_mask
        # output:
        #   self.peduncle_orienation: (x,y,z)
        #################################################################
        pass


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

