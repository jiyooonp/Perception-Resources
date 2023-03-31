#!/usr/bin/env python3
import rospy 
from pipeline import Perception

if __name__ == '__main__':
    # img_path = '../dataset/testbed_video_to_img'
    img_path = '/test'
    pipeline = Perception(img_path, 0)
    # pipeline.detect_peppers_in_folder()
    pipeline.send_to_manipulator()

'''
input an image
    make a one_frame
        run pepper_fruit detection
        run pepper_peduncle detection
        match pepper
    get peduncle location 

'''
