import os
import shutil
import random
import ultralytics
ultralytics.checks()
os.system('yolo task=detect mode=predict model=./weights/best.pt conf=0.55 source=0 save=True')
