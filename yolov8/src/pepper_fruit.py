class PepperFruit:
    def __init__(self, number:int, order = None, x=None, y=None, w=None, h=None, conf=None):
        self.number = number

        self._xywh = list()
        self._conf = conf
        self._order = order

    @property
    def xywh(self):
        return self._xywh
    @xywh.setter
    def xywh(self, xywh):
        self._xywh = xywh
    @property
    def conf(self):
        return self._conf
    @conf.setter
    def conf(self, conf):
        self._conf = conf
    @property
    def order(self):
        return self._order
    @order.setter
    def order(self, order):
        self._order = order

    def __str__(self):
        return f"Pepper(number={self.number}, order={self._order}, xywh={self.xywh}, conf={self._conf})"
