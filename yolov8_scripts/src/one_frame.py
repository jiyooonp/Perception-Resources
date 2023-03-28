from typing import Optional, Tuple, List, Dict

from yolov8_scripts.src import pepper_utils
from yolov8_scripts.src.pepper_fruit import PepperFruit
from yolov8_scripts.src.pepper_fruit_detector import PepperFruitDetector
from yolov8_scripts.src.pepper_peduncle import PepperPeduncle
from yolov8_scripts.src.pepper import Pepper
from yolov8_scripts.src.pepper_peduncle_detector import PepperPeduncleDetector
from yolov8_scripts.src.pepper_utils import *
import torch
class OneFrame:
    def __init__(self, img_path):
        self.img_path = img_path # should be a path to one image file

        self._img_shape: Tuple[int] = pepper_utils.get_img_size(img_path)

        self._mask: Optional[torch.Tensor] = None

        self.detected_pepper_fruit: bool = False
        self.detected_pepper_peduncle: bool = False

        self._pepper_fruit_count: int = 0
        self._pepper_peduncle_count: int = 0

        self._pepper_fruit_detections: Dict[int, PepperFruit] = dict()
        self._pepper_peduncle_detections: Dict[int. PepperPeduncle] = dict()
        self._pepper_detections: Dict[int, Pepper] = dict()

        self._pepper_fruit_detector: PepperFruitDetector = PepperFruitDetector(img_path,
           yolo_weight_path = '/home/jy/PycharmProjects/Perception-Resources/yolov8_scripts/weights/pepper_fruit_best_2.pt')
        self._pepper_peduncle_detector: PepperPeduncleDetector = PepperPeduncleDetector(img_path,
            yolo_weight_path = '/home/jy/PycharmProjects/Perception-Resources/yolov8_scripts/weights/pepper_peduncle_best.pt')

    @property
    def img_shape(self):
        return self._img_shape
    @property
    def mask(self):
        return self._mask
    @mask.setter
    def mask(self, mask):
        self._mask = mask
    @property
    def pepper_fruit_detections(self):
        return self._pepper_fruit_detections
    @property
    def pepper_peduncle_detections(self):
        return self._pepper_peduncle_detections
    @property
    def pepper_fruit_count(self):
        return self._pepper_fruit_count
    @property
    def pepper_peduncle_count(self):
        return self._pepper_peduncle_count
    @property
    def pepper_detections(self):
        return self._pepper_detections
    def __str__(self):
        return f"DetectedFrame(frame={self.frame}, detections={self.detections})"

    def add_detected_pepper_fruit(self, pepper):
        self._pepper_fruit_detections[pepper.number] = pepper
        self._pepper_fruit_count += 1
    def add_detected_pepper_peduncle(self, peduncle):
        self._pepper_peduncle_detections[peduncle.number] = peduncle
        self._pepper_peduncle_count += 1
    def match_peppers(self):
        pepper_fruit_peduncle_match= match_pepper_fruit_peduncle(self._pepper_fruit_detections, self._pepper_peduncle_detections)
        number = 0
        for (pfn, ppn), _ in pepper_fruit_peduncle_match:
            if ppn == -1:
                continue
            else:
                pepper = Pepper(number, pfn, ppn)
                pepper.pepper_fruit = self._pepper_fruit_detections[pfn]
                pepper.pepper_peduncle = self.pepper_peduncle_detections[ppn]
                self._pepper_detections[number] = pepper
                number += 1
    def plot_pepper_fruit(self):
        draw_pepper_fruits(self)
    def plot_pepper_peduncle(self):
        draw_pepper_peduncles(self)
    def plot_pepper(self):
        draw_pepper(self)
    def run(self):
        self._pepper_fruit_detections = self._pepper_fruit_detector.run_detection(self.img_path, thresh=0.3, show_result=False)
        self._pepper_peduncle_detections = self._pepper_peduncle_detector.run_detection(self.img_path,thresh=0.3, show_result=False)
        print(self._pepper_peduncle_detections)
        self.match_peppers()
        self.plot_pepper()


