from shapely.geometry import Polygon

from typing import List

from yolov8_scripts.src.one_frame import OneFrame
from yolov8_scripts.src.pepper import Pepper


class MultiFrame:
    def __init__(self):
        self._one_frames: List[OneFrame] = list()
        self._positive_peppers: List[Pepper] = list()



box_1 = [[511, 41], [577, 41], [577, 76], [511, 76]]
box_2 = [[544, 59], [610, 59], [610, 94], [544, 94]]

print(calculate_iou(box_1, box_2))