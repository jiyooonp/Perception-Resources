from pipeline import Perception

if __name__=='__main__':
    # img_path = '/home/jy/PycharmProjects/Perception-Resources/dataset/peduncle/pepper-plant-OPT-600x600.jpg'
    # img_path = '/home/jy/PycharmProjects/Perception-Resources/dataset/peduncle/img_4223.jpg'
    img_path = '/home/jy/PycharmProjects/Perception-Resources/dataset/peduncle'
    # img_path = '/home/jy/PycharmProjects/Perception-Resources/dataset/colorful'
    pipeline = Perception(img_path, 0)
    pipeline.detect_peppers_in_folder()


'''
input an image
    make a one_frame
        run pepper_fruit detection
        run pepper_peduncle detection
        match pepper
    get peduncle location 

'''

