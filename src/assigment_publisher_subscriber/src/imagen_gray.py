#!/usr/bin/env python

import numpy as np
import rospy
import cv2
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

        gray=cv2.cvtColor(self.net_image, cv2.COLOR_BGR2GRAY)
        bi_gray_max = 255
        bi_gray_min = 70
        ret,thresh1=cv2.threshold(gray, bi_gray_min, bi_gray_max, cv2.THRESH_BINARY)

        test_images = thresh1
	print("Tamanio imagen: {}".format(test_images.shape))
	test_images = cv2.resize(test_images, (28, 28), interpolation = cv2.INTER_AREA) 
        #test_images = test_images.reshape((28,28))
        #test_images = test_images.astype('float32') / 255


        try:
            ros_img = self.bridge.cv2_to_imgmsg(test_images)
            self.img_pub.publish(ros_img)
        except CvBridgeError as e:
            print(e)




if __name__ == '__main__':
    try:
        rospy.init_node('parking')
        dummy = parking() # Crea un objeto para ejecutar parking
        rate = rospy.Rate(10) # Que tan rapido quiero que se ejecute el programa
        while not rospy.is_shutdown():
            dummy.simple_parking()
            rate.sleep()
    except rospy.ROSInterruptException:
        print("Algo raro esta pasando")
        pass
