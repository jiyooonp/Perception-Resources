from typing import Optional, Tuple

import pepper_utils
from pepper import Pepper
from pepper_fruit_detector import PepperFruitDetector
from pepper_peduncle_detector import PepperPeduncleDetector
from pepper_peduncle_utils import *
from pepper_utils import *
import os


class OneFrame:
    def __init__(self, img_path):
        self.img_path = img_path  # should be a path to one image file

        self._img_shape: Tuple[int] = pepper_utils.get_img_size(img_path)

        self._mask: Optional[torch.Tensor] = None

        self.detected_pepper_fruit: bool = False
        self.detected_pepper_peduncle: bool = False

        self._pepper_fruit_count: int = 0
        self._pepper_peduncle_count: int = 0

        self._pepper_fruit_detections: Dict[int, PepperFruit] = dict()
        self._pepper_peduncle_detections: Dict[int.PepperPeduncle] = dict()
        self._pepper_detections: Dict[int, Pepper] = dict()

        print("weights path: ", os.getcwd())
        self._pepper_fruit_detector: PepperFruitDetector = PepperFruitDetector(img_path,
                                                                               yolo_weight_path=os.getcwd()+'/pepper_ws/weights/pepper_fruit_best_2.pt')
        self._pepper_peduncle_detector: PepperPeduncleDetector = PepperPeduncleDetector(img_path,
                                                                                        yolo_weight_path=os.getcwd()+'/pepper_ws/weights/pepper_peduncle_best.pt')

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
        pepper_fruit_peduncle_match = match_pepper_fruit_peduncle(self._pepper_fruit_detections,
                                                                  self._pepper_peduncle_detections)
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

    def determine_peduncle_poi(self):
        for key, single_pepper in self._pepper_detections.items():
            single_pepper.pepper_peduncle.set_point_of_interaction(self._img_shape, single_pepper.pepper_fruit.xywh)

    def determine_peduncle_orientation(self):
        for key, single_pepper in self._pepper_detections.items():
            single_pepper.pepper_peduncle.set_peduncle_orientation(single_pepper.pepper_fruit.xywh)

    def determine_pepper_order(self, arm_xyz):
        pepper_distances = {}
        for _, pepper in self.pepper_detections.items():
            poi = pepper.pepper_peduncle.poi
            dist = np.linalg.norm(poi - arm_xyz)
            pepper_distances[dist] = pepper

        distances = list(pepper_distances.keys()).sort()
        order = 1
        for i in distances:
            pepper = pepper_distances[i]
            pepper.order = order
            order += 1

    def plot_pepper_fruit(self):
        draw_pepper_fruits(self)

    def plot_pepper_peduncle(self):
        draw_pepper_peduncles(self)

    def plot_pepper(self):
        draw_pepper(self)

    def plot_poi(self):
        draw_poi(self)

    def run(self):
        self._pepper_fruit_detections = self._pepper_fruit_detector.run_detection(self.img_path, thresh=0.3,
                                                                                  show_result=False)
        print("what ever", self._pepper_fruit_detections )
        self.plot_pepper_fruit()
        self._pepper_peduncle_detections = self._pepper_peduncle_detector.run_detection(self.img_path, thresh=0.3,
                                                                                        show_result=False)
        self.plot_pepper_peduncle()
        self.match_peppers()
        self.plot_pepper()
        

        self.determine_peduncle_poi()
        # self.plot_poi()
