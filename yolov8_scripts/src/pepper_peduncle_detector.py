from typing import NamedTuple, Tuple, List

import cv2
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
import ultralytics
# from detected_frame import DetectedFrame
# from pepper_peduncle import PepperPeduncle
# from pepper_utils import *

from yolov8_scripts.src.detected_frame import DetectedFrame
from yolov8_scripts.src.pepper_peduncle import PepperPeduncle
from yolov8_scripts.src.pepper_utils import *


# class PepperPeduncleBatch:
#     bbox: torch.Tensor
#     mask: torch.Tensor
#
#
# class PepperBatch(NamedTuple):
#     img: torch.Tensor
#     mask: torch.Tensor
#     pepper_peduncle_batch: PepperPeduncleBatch
#     @classmethod
#     def from_tuple(cls, pepper_batch_tuple: Tuple):
#         cls(pepper_batch_tuple[0], pepper_batch_tuple[1])
#
#
# class PepperDataset:
#     def __init__(self):
#         pass
#
#     def __getitem__(self):
#         return PepperBatch(torch.rand(3, 256, 256), torch.rand(3, 256, 256))
#
#
#
# dataloader = DataLoader(PepperDataset())
# batch: Tuple = next(iter(dataloader))
# pepper_batch = PepperBatch.from_tuple(batch)
#
# pepper_batch.pepper_peduncle_batch.bbox


class PepperPeduncleDetector:
    def __init__(self, file_path, yolo_weight_path):

        ultralytics.checks()

        self._model: YOLO = YOLO(yolo_weight_path)
        self._path: str = file_path
        self._classes: List[str] = ["pepper"]

        self._imgs_path: List[str] = list()
        self._detected_frames: List[DetectedFrame] = list()
    @property
    def detected_frames(self):
        return self._detected_frames
    @detected_frames.setter
    def detected_frames(self, detected_frames):
        self._detected_frames = detected_frames
    @property
    def path(self):
        return self._path
    def __str__(self):
        return print_pepperdetection(self)
    def run_detection(self, show_result: bool = False, print_result: bool = False):
        print("Starting detection")
        self._imgs_path = get_all_image_path_in_folder(self._path)
        self.predict_peduncles(show_result, print_result)

    def predict_peduncle(self, img_path, show_result: bool = False, print_result: bool = False):
        detected_frame = DetectedFrame(img_path)
        print("Detecting image: ", img_path)

        img = read_image(img_path)
        results = self._model(img)
        peduncle_count = 0

        result = results[0]

        for i in range(result.masks.shape[0]):
            mask = result.masks # Boxes object for bbox outputs
            box = result.boxes

            peduncle = PepperPeduncle(peduncle_count)

            peduncle.mask = torch.Tensor(mask.segments[i])
            peduncle.conf = box.conf[i]

            detected_frame.img_shape = img.shape
            detected_frame.add_detected_pepper_peduncle(peduncle)
            peduncle_count += 1
        detected_frame.mask = result.masks.masks
        if show_result:
            for result in results:
                res_plotted = result[0].plot()
                # cv2.imshow("result", res_plotted)

        if print_result:
            print_result_masks(detected_frame)

        return detected_frame
    def predict_peduncles(self, show_result: bool = False, print_result: bool = False):

        for img_path in self._imgs_path:
            detected_frame = self.predict_peduncle(img_path, show_result, print_result)
            self._detected_frames.append(detected_frame)
    def plot_results(self):
        print("Plotting results")
        for detected_img in self.detected_frames:
            draw_peduncles(detected_img)


if __name__ == '__main__':
    PepperPeduncleDetection = PepperPeduncleDetector(file_path='/home/jy/PycharmProjects/Perception-Resources/dataset/peduncle', yolo_weight_path="../weights/pepper_peduncle_best.pt")
    PepperPeduncleDetection.run_detection()
    print(PepperPeduncleDetection)
    PepperPeduncleDetection.plot_results()
