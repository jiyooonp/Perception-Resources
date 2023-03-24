from typing import Optional, Tuple, List
from yolov8_scripts.src.pepper_fruit import PepperFruit
from yolov8_scripts.src.pepper_peduncle import PepperPeduncle
import torch
class DetectedFrame:
    def __init__(self, img_path):
        self.img_path = img_path

        self._img_shape: Optional[Tuple[int]] = (0, 0)
        self._mask: Optional[torch.Tensor] = None
        self.detected_pepper_fruit: bool = False
        self.detected_peduncle_fruit: bool = False
        self._pepper_fruit_count: int = 0
        self._pepper_peduncle_count: int = 0
        self._pepper_fruit_detections: List[PepperFruit] = list()
        self._pepper_peduncle_detections: List[PepperPeduncle] = list()
    @property
    def img_shape(self):
        return self._img_shape
    @img_shape.setter
    def img_shape(self, img_shape):
        self._img_shape = img_shape
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
    def __str__(self):
        return f"DetectedFrame(frame={self.frame}, detections={self.detections})"

    def add_detected_pepper_fruit(self, pepper):
        self._pepper_fruit_detections.append(pepper)
        self._pepper_fruit_count += 1
    def add_detected_pepper_peduncle(self, peduncle):
        self._pepper_peduncle_detections.append(peduncle)
        self._pepper_peduncle_count += 1
