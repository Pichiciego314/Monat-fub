#!/usr/bin/env python

import numpy as np
import rospy
import tf
import rospkg
from std_msgs.msg import Float64, Header
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class parking:
    def __init__(self):
        self.net_image = []
        self.permanente = 0  # Variable de vida permanente
        self.camara_sub = rospy.Subscriber("/sensors/camera/color/image_raw", Image, self.get_image)
        self.img_pub = rospy.Publisher("/imagen/nuevo_topico", Image, queue_size=10)
        self.bridge = CvBridge()

    def get_image(self, mensaje_recibido):
        try:
            self.net_image = self.bridge.imgmsg_to_cv2(mensaje_recibido)
        except CvBridgeError as e:
            print(e)

    def simple_parking(self):
        if self.net_image == []:
            return

        try:
            ros_img = self.bridge.cv2_to_imgmsg(self.net_image)
            self.img_pub.publish(ros_img)
        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':
    try:
        rospy.init_node('imagen')
        dummy = parking() # Crea un objeto para ejecutar parking
        rate = rospy.Rate(10) # Que tan rapido quiero que se ejecute el programa
        while not rospy.is_shutdown():
            dummy.simple_parking()
            rate.sleep()
    except rospy.ROSInterruptException:
        print("Algo raro esta pasando")
        pass
