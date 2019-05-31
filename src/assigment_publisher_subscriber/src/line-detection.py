#!/usr/bin/env python
import roslib
import sys
import cv2
import numpy as np
import rospy
import tf
import rospkg
from std_msgs.msg import Float64, Header
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#imagen para ser capturada
cap = cv2.VideoCapture(0)

class linea:
    def __init__(self):
        self.net_image = []
        self.lidar = rospy.Subscriber("/sensors/rplidar/scan", Image, self.detectar)
        self.camara_subs = rospy.Subscriber("/sensors/camera/color/image_raw", Image,self.get_image)
        self.image_pub = rospy.Publisher("/imagen/nuevo_topico", Image, queue_size=10)
        self.bridge = CvBridge()

    def get_image(self, recived_message):
        try:
            self.net_image = self.bridge.imgmsg_to_cv2(recived_message)
        except CvBridgeError as e:
            print(e)

    def detectar(self):
        if self.net_image == []:
            return
#convierto en HVS y 

        _, self.net_image = cap.read()

        hsv = cv2.cvtColor(self.net_image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0,0,0], dtype=np.uint8)
        upper_white = np.array([0,0,255], dtype=np.uint8)
        inter = cv2.inRange(hsv, lower_white, upper_white)
        lasts = cv2.bitwise_and(self.net_image,self.net_image, mask= inter)

        cv2.imshow('self.net_image',self.net_image)
        cv2.imshow('inter',inter)
        cv2.imshow('lasts',lasts)
        k = cv2.waitKey(5) & 0xFF


        cv2.destroyAllWindows()

        try:
            ros_img = self.bridge.cv2_to_imgmsg(lasts)
            self.image_pub.publish(ros_img)
        except CvBridgeError as e:
            print(e)

        
 
if __name__ == '__main__':
    try:
        rospy.init_node('line-detection')
        dummy = linea() # Crea un objeto para ejecutar parking
        rate = rospy.Rate(10) # Que tan rapido quiero que se ejecute el programa
        while not rospy.is_shutdown():
            dummy.detectar()
            rate.sleep()
    except rospy.ROSInterruptException:
        print("Algo raro esta pasando")
        pass
