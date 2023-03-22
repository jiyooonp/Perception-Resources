class DetectedFrame:
    def __init__(self, img_path):
        self.img_path = img_path
        self.detect_complete = False
        self.pepper_count = 0
        self.detections = list()

    def __str__(self):
        return f"DetectedFrame(frame={self.frame}, detections={self.detections})"

    # def __repr__(self):
    #     return self.__str__()

    # settter methods
    def set_img_path(self, img_path):
        self.img_path = img_path
    def set_pepper_count(self, count):
        self.pepper_count = count
    def add_detected_pepper(self, Pepper):
        self.detections.append(Pepper)
        self.pepper_count += 1

    # getter methods
    def get_img_path(self):
        return self.img_path
    def get_pepper_count(self):
        return self.pepper_count
    def get_detections(self):
        return self.detections
