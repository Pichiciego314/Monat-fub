#!/usr/bin/env python
from keras import models, layers
from keras.utils import to_categorical
import keras
import cv2
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
        self.camara_sub = rospy.Subscriber("/sensors/camera/color/image_raw", Image, self.get_image)
        self.img_pub = rospy.Publisher("/imagen/nuevo_topico", Image, queue_size=10)
        self.bridge = CvBridge()
        self.network = models.Sequential()
        self.network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
        self.network.add(layers.Dense(10, activation='softmax'))
        self.network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    def get_image(self, mensaje_recibido):
        try:
            self.net_image = self.bridge.imgmsg_to_cv2(mensaje_recibido)
        except CvBridgeError as e:
            print(e)

    def simple_parking(self):
        if self.net_image == []:
            return


        self.network.load_weights('/home/alvarado/catkin_emiliano/src/assigment_publisher_subscriber/src/trained.h5')

        gray=cv2.cvtColor(self.net_image, cv2.COLOR_BGR2GRAY)
        #bi_gray_max = 255
        #bi_gray_min = 70
        #ret,thresh1=cv2.threshold(gray, bi_gray_min, bi_gray_max, cv2.THRESH_BINARY)
        ret2,thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        test_images = thresh1
	print("Tamanio imagen: {}".format(test_images.shape))
	test_images = cv2.resize(test_images, (28, 28), interpolation = cv2.INTER_AREA) 

        try:
            ros_img = self.bridge.cv2_to_imgmsg(test_images)
            self.img_pub.publish(ros_img)
        except CvBridgeError as e:
            print(e)

        imagenes_NN = test_images.reshape((1,28*28))
        imagenes_NN = imagenes_NN.astype('float32') / 255

#cv2.imshow('mnist', imagen_test)
#cv2.waitKey(0)

        resultado = self.network.predict_classes(imagenes_NN)
 
        print("Resultado: {}".format(resultado))



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
