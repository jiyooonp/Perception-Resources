from pipeline import Perception

if __name__=='__main__':
    img_path = '/home/jy/PycharmProjects/Perception-Resources/dataset/testbed_video_to_img'
    pipeline = Perception(img_path, 0)
    pipeline.detect_peppers_in_folder()
    pipeline.send_to_manipulator()


'''
input an image
    make a one_frame
        run pepper_fruit detection
        run pepper_peduncle detection
        match pepper
    get peduncle location 

'''

