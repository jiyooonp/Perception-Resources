import matplotlib.pyplot as plt
from ultralytics import YOLO
import ultralytics
from yolov8_scripts.src.detected_frame import DetectedFrame
from yolov8_scripts.src.pepper_fruit import PepperFruit
from yolov8_scripts.src.pepper_utils import *
from typing import List

class PepperFruitDetector:
    def __init__(self, file_path: str, yolo_weight_path: str):

        ultralytics.checks()

        self._model: YOLO = YOLO(yolo_weight_path)
        self._path: str = file_path
        self._classes: List[str] = ["pepper"]

        self._imgs_path: List[str] = list()
        self._detected_frames: List[DetectedFrame] = list()
    @property
    def detected_frames(self):
        return self._detected_frames
    @property
    def classes(self):
        return self._classes

    def __str__(self):
        return print_pepperdetection(self)

    def run_detection(self, show_result: bool = False, print_result: bool = False):
        print("Starting detection")
        self._imgs_path = get_all_image_path_in_folder(self._path)
        self.predict_peppers(show_result, print_result)

    def predict_pepper(self, img_path, show_result: bool = False, print_result: bool = False):
        detected_frame = DetectedFrame(img_path)
        print("Detecting image: ", img_path)

        img = read_image(img_path)
        results = self._model(img)
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
                detected_frame.add_detected_pepper_fruit(pepper)
                pepper_count += 1

        # UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure. plt.show()
        if show_result:
            for result in results:
                res_plotted = result.plot()
                # cv2.imshow("result", res_plotted)
        if print_result:
            print_result_boxes(detected_frame)
        return detected_frame

    def predict_peppers(self, show_result: bool = False, print_result: bool = False):

        for img_path in self._imgs_path:
            detected_frame = self.predict_pepper(img_path, show_result, print_result)
            self._detected_frames.append(detected_frame)
    def plot_results(self):
        print("Plotting results")
        cnt = 0
        for detected_img in self._detected_frames:
            cnt +=1
            print("Plotting image: ", cnt)
            draw_peppers(detected_img)

    def webcam_prediction(self):
        cam = cv2.VideoCapture(0)
        cv2.namedWindow('Realtime Image')
        img_counter = 0
        while True:
            ret, image = cam.read()
            if not ret:
                print('failed to grab frame')
                break
            k = cv2.waitKey(1)
            if k % 256 == 27:
                print('escape hit, closing the app')
                break
            # screenshots will be taken
            elif k % 256 == 32:
                # the format for storing the images scrreenshotted
                img_name = f'opencv_frame_{img_counter}.jpg'
                # saves the image as a png file
                cv2.imwrite(img_name, image)
                print('screenshot taken')
                # the number of images automaticallly increases by 1
                img_counter += 1

            results = self._model.predict(source=image, save=True, save_txt=True)

            if results[0].probs:
                print(f"detected {len(results)} objects!")
                for result in results:
                    boxes = result.boxes  # Boxes object for bbox outputs
                    image = result.orig_img
                    for box in boxes:
                        print(f"detected {[box.cls]}")
                        x, y, w, h = box._xywh
                        self.draw_bounding_box(image, box.cls, box._conf, x, y, x + w, y - h)
            else:
                print(f"detected nothing")
                cv2.putText(image, "nothing found", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow('Input Image', image)
        cam.release()
        cam.destoryAllWindows()

if __name__ == '__main__':
    PepperDetection = PepperFruitDetector(file_path='/home/jy/PycharmProjects/Perception-Resources/dataset/testbed_video_to_img', yolo_weight_path="../weights/pepper_fruit_best.pt")
    PepperDetection.run_detection(show_result=False)
    print(PepperDetection)
    PepperDetection.plot_results()
