from ultralytics import YOLO
import ultralytics
from yolov8_scripts.src.pepper_fruit import PepperFruit
from yolov8_scripts.src.pepper_utils import print_pepperdetection, get_all_image_path_in_folder,read_image, draw_pepper_fruits, red_to_green_2
from typing import List

class PepperFruitDetector:
    def __init__(self, file_path: str, yolo_weight_path: str):

        ultralytics.checks()

        self._model: YOLO = YOLO(yolo_weight_path)
        self._path: str = file_path
        self._classes: List[str] = ["pepper"]

        self._imgs_path: List[str] = list()
        # self._detected_frames: List[OneFrame] = list()
    @property
    def detected_frames(self):
        return self._detected_frames
    @property
    def classes(self):
        return self._classes

    def __str__(self):
        return print_pepperdetection(self)

    def run_detection(self, img_path, show_result: bool = False, print_result: bool = False, thresh = 0.25):
        # print("Starting detection")
        self._imgs_path = get_all_image_path_in_folder(self._path)
        return self.predict_pepper(img_path, show_result, print_result, thresh=thresh)


    def predict_pepper(self, img_path, show_result: bool = False, print_result: bool = False, thresh=0.25):
        pepper_list = dict()
        print("Detecting image: ", img_path)

        img = read_image(img_path)
        # img = red_to_green_2(img).astype('float32')
        results = self._model(img, conf=thresh)
        pepper_count = 0

        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            for box in boxes:
                one_box = box[0]
                pepper = PepperFruit(pepper_count)
                xywh = one_box.xywh
                conf = one_box.conf  # Class probabilities for classification outputs
                pepper.xywh = xywh[0].cpu().numpy()
                pepper.conf = conf
                pepper_list[pepper_count] = pepper
                pepper_count += 1

        # UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure. plt.show()
        if show_result:
            for result in results:
                res_plotted = result.plot()
                # cv2.imshow("result", res_plotted)
        # if print_result:
        #     print_result_boxes(detected_frame)
        return pepper_list

    def predict_peppers(self, show_result: bool = False, print_result: bool = False):

        for img_path in self._imgs_path:
            detected_frame = self.predict_pepper(img_path, show_result, print_result)
            self._detected_frames.append(detected_frame)
    def plot_results(self):
        print("Plotting results")
        cnt = 0
        for detected_img in self._detected_frames:
            cnt +=1
            draw_pepper_fruits(detected_img)

if __name__ == '__main__':
    PepperDetection = PepperFruitDetector(file_path='/home/jy/PycharmProjects/Perception-Resources/dataset/peduncle', yolo_weight_path="../weights/pepper_fruit_best_2.pt")
    PepperDetection.run_detection(img_path='/home/jy/PycharmProjects/Perception-Resources/dataset/peduncle', show_result=False)
    print(PepperDetection)
    PepperDetection.plot_results()
