import rospy
from geometry_msgs.msg import Pose

class Communication:
    def __init__(self):
        self.poi_pub = rospy.Publisher('/perception/peduncle/poi', Pose, queue_size=10)

    def poi_pub(self, poi, orientation):
        peduncle_pose = Pose()
        peduncle_pose.position = poi
        # peduncle_pose.orientation = orientation
        rospy.loginfo(peduncle_pose)
        self.poi_pub.publish(peduncle_pose)
