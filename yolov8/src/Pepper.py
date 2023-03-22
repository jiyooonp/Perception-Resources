class Pepper:
    def __init__(self, number:int, order = None, x=None, y=None, w=None, h=None, conf=None):
        self.number = number

        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.conf = conf
        self.order = order
    def __str__(self):
        return f"Pepper(number={self.number}, order={self.order}, x={self.x}, y={self.y}, w={self.w}, h={self.h}, conf={self.conf})"
    # setter methods
    def set_xywh(self, xywh):
        self.x = xywh[0]
        self.y = xywh[1]
        self.w = xywh[2]
        self.h = xywh[3]
    def set_conf(self, conf):
        self.conf = conf
    def set_order(self, order):
        self.order = order

    # getter methods
    def get_number(self):
        return self.number
    def get_xywh(self):
        return (self.x, self.y, self.w, self.h)
    def get_conf(self):
        return round(self.conf[0].item(), 2)