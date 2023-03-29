import numpy as np
from skimage.morphology import medial_axis
from yolov8_scripts.src.curve import Curve

class PepperPeduncle:
    def __init__(self, number:int, mask = None, conf=None):
        self.number: int = number
        self._mask = mask
        self._conf: float = conf
        self._xywh = None
        self._curve = Curve()
        self._poi = None

    @property
    def mask(self):
        return self._mask
    @mask.setter
    def mask(self, mask):
        self._mask = mask
    @property
    def conf(self):
        if self._conf is None:
            return 0
        return self._conf
    @conf.setter
    def conf(self, conf):
        self._conf = conf
    @property
    def xywh(self):
        return self._xywh
    @xywh.setter
    def xywh(self, value):
        self._xywh = value
    @property
    def curve(self):
        return self._curve
    @curve.setter
    def curve(self, curve):
        self._curve = curve
    @property
    def poi(self):
        return self._poi
    @poi.setter
    def poi(self, value):
        self._poi = value
    def __str__(self):
        return f"Peduncle(number={self.number},mask={self._mask}, conf={self._conf})"
