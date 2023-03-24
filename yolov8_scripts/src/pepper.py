from yolov8_scripts.src.pepper_peduncle import PepperPeduncle
from yolov8_scripts.src.pepper_fruit import PepperFruit
class Pepper:
    def __init__(self, number:int):
        self._number = number

        self._pepper_fruit: PepperFruit = PepperFruit(number)
        self._pepper_peduncle: PepperPeduncle = PepperPeduncle(number)

    @property
    def number(self):
        return self._number
    @property
    def pepper_fruit(self):
        return self._pepper_fruit
    @property
    def pepper_peduncle(self):
        return self._pepper_peduncle
    def __str__(self):
        return f"Pepper #{self.number}"

