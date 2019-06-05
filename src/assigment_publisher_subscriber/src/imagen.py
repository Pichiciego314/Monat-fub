#!/usr/bin/env python
# Here the libraries are imported
import numpy as np
import rospy
import tf
import rospkg
from std_msgs.msg import Float64, Header
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#here "image" is captured
class image:
    #Here are the permanent variables
    def __init__(self):
        self.net_image = []
        self.camara_sub = rospy.Subscriber("/sensors/camera/color/image_raw", Image, self.get_image)#here is the subscription to the camera topic
        self.img_pub = rospy.Publisher("/imagen/nuevo_topico", Image, queue_size=10)#I publish the new image into a new topic
        self.bridge = CvBridge()#the funcition that will convert the images

    def get_image(self, recived_message):
        try:
            self.net_image = self.bridge.imgmsg_to_cv2(recived_message)
        except CvBridgeError as e:
            print(e)

    def image_converter(self):
        if self.net_image == []:
            return

        try:
            #here are the convertions of the image into something the computer can read
            ros_img = self.bridge.cv2_to_imgmsg(self.net_image)
            self.img_pub.publish(ros_img)
        except CvBridgeError as e:
            print(e)

#definition of main function
if __name__ == '__main__':
    try:
        rospy.init_node('imagen')# node that is going to be initialize
        dummy = image() # creates object to execute parking
        rate = rospy.Rate(10) # Que tan rapido quiero que se ejecute el programa
        while not rospy.is_shutdown():
            dummy.image_converter()
            rate.sleep()
    except rospy.ROSInterruptException:
        print("Algo raro esta pasando")
        pass
