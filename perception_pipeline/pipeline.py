import time
# input: image
class Perception:
    def __init__(self, source, fps, save=True):
        self.source = source
        self.start_time = time.now()
        self.fps = fps
        self.save = save
    def get_image(self):
        #################################################################
        # get image from source
        # output: RGBD information
        #################################################################
        self.image = None
        pass
    def get_depth(self, image, x, y):
        #################################################################
        # given an image and x, y coordinates, return the depth information
        # note this will take in an area in the image, average the depth
        # and do additional calculation to compensate for the noise given
        # by the RGBD camera
        # input:
        #   image: (H, W, D)?
        #   x, y: coordinates
        # output:
        #   d: depth of that point
        #################################################################
        pass
    def process_image(self):
        #################################################################
        # process the image to match the dim for yolo (prob don't need this)
        #################################################################
        pass

    #####################################################################
    # When base goes to one location, use the long range images and retreive
    # all the locations of peppers
    #####################################################################
    def detect_peppers_one_frame(self, thresh):
        #################################################################
        # use yolov8 and get the pepper locations
        # input:
        #   thresh: disgard detection lower than threshold
        # output:
        #   locations: all the locations of the pepper boxes [conf, x, y] (N, 3)
        #################################################################
        pass
    def detect_peppers_time_frame(self, thresh, frames):
        #################################################################
        # stack pepper locations over a timeframe time
        # input:
        #   frames: F number of frames to be stored in list
        # output:
        #   locations: all the locations of the pepper boxes over a
        #       number of frames F x [conf, x, y] (F, N, 3)
        #################################################################
        pass
    def clear_false_positives(self):
        #################################################################
        # algorithm to tak in a series of bounding boxes and output
        # true positive pepper locations
        # input:
        #   locations : pepper locations over F frames (F, N, 3)
        # output:
        #   self.pepper_locations: true positive pepper locations including
        #   the depth information (M, 4)
        #################################################################
        pass

    #####################################################################
    # Once the manipulator goes closer to the pepper, we have one pepper
    # as target.
    #####################################################################
    def set_target_pepper(self, pepper_index):
        #################################################################
        # Using the pepper_index, take another closer image of the pepper,
        # run the detection algorithm to get a more precise bounding box.
        # Store the pepper's information in self.pepper
        # output:
        #   self.pepper = {"idx": None, "box": (L, T, R, D), "location": (xc, yc, d)}
        #################################################################
        pass
    def get_peduncle_location(self):
        #################################################################
        # for self.pepper, crop the image, run the segmentation model and
        # get the segmented peduncle mask.
        # output:
        #   self.peduncle_mask: idk what this form is
        #################################################################
        pass
    def get_point_of_interaction(self):
        #################################################################
        # using self.peduncle_mask, calculate the point of interaction
        # input:
        #   self.peduncle_mask
        # output:
        #   self.poi: (x, y, d)
        #################################################################
        pass
    def get_peduncle_orientation(self):
        #################################################################
        # calculate the orientation of the peduncle using self.peduncle_mask
        # output:
        #   self.peduncle_orienation: (x,y,z)
        #################################################################
        pass


    #####################################################################
    # ROS related
    #####################################################################


    def send_to_manipulator(self):
        #################################################################
        # send the point of interaction to the manipulator over ROS
        #################################################################
        pass


    #####################################################################
    # VISUALIZATION related
    #####################################################################

    
    def send_to_gui(self):
        #################################################################
        # send information to gui over ros
        #################################################################
        pass
    def get_from_gui(self):
        #################################################################
        # get information from gui over ros
        # such as commands (stop running/change fps/etc)
        #################################################################
        pass

