from typing import List, Optional
class PepperFruit:
    def __init__(self, number:int, order = -1, xywh=None, conf=0.0):
        self._number: int = number

        self._xywh: Optional[List[float]] = xywh
        self._order: int = order
        self._conf: float = conf
    @property
    def number(self):
        return self._number
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
