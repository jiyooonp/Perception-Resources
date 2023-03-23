from pepper_peduncle import PepperPeduncle
from pepper_fruit import PepperFruit
class Pepper:
    def __init__(self, number:int, mask = None, conf=None):
        self._pepper_fruit = PepperFruit(number)
        self._pepper_peduncle = PepperPeduncle(number)

    @property
    def pepper_fruit(self):
        return self._pepper_fruit
    @pepper_fruit.setter
    def pepper_fruit(self, pepper_fruit):
        self._pepper_fruit = pepper_fruit
    @property
    def pepper_peduncle(self):
        return self._pepper_peduncle
    @pepper_peduncle.setter
    def pepper_peduncle(self, pepper_peduncle):
        self._pepper_peduncle = pepper_peduncle
    def __str__(self):
        return f"Pepper #{self.number}"
