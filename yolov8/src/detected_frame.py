class DetectedFrame:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_shape = None
        self.detected_pepper_fruit = False
        self.detected_peduncle_fruit = False
        self._pepper_fruit_count = 0
        self._pepper_peduncle_count = 0
        self._pepper_fruit_detections = list()
        self._pepper_peduncle_detections = list()

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

    def add_detected_pepper_fruit(self, Pepper):
        self._pepper_fruit_detections.append(Pepper)
        self._pepper_fruit_count += 1
    def add_detected_pepper_peduncle(self, Peduncle):
        self._pepper_peduncle_detections.append(Peduncle)
        self._pepper_peduncle_count += 1
