#!/usr/bin/env python
from keras import models, layers
from keras.utils import to_categorical
import keras
import cv2
import numpy as np
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dense, Dropout, Flatten
import rospy
import tf
import rospkg
from std_msgs.msg import Float64, Header
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from autominy_msgs.msg import SteeringAngle, NormalizedSpeedCommand
from std_msgs.msg import String



class parking:
    def __init__(self):
        self.net_image = []
        self.camara_sub = rospy.Subscriber("/sensors/camera/color/image_raw", Image, self.get_image)
        self.img_pub = rospy.Publisher("/imagen/nuevo_topico", Image, queue_size=10)
        self.bridge = CvBridge()
        self.vel_pub = rospy.Publisher("/control/command/normalized_wanted_speed", NormalizedSpeedCommand, queue_size=10)
        self.network = models.Sequential()
        self.network.add(layers.Dense(512, activation='relu', input_shape= (28*28,)))
        self.network.add(layers.Dense(6, activation='softmax'))
        self.network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        #estructura de la red convoluional
        classifier = models.Sequential()
        classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
        BatchNormalization(axis=-1)  # Axis -1 is always the features axis
        classifier.add(Activation('relu'))
        classifier.add(Conv2D(64, (3, 3)))
        BatchNormalization(axis=-1)
        classifier.add(Activation('relu'))
        classifier.add(MaxPool2D(pool_size=(2, 2)))
        BatchNormalization(axis=-1)
        classifier.add(Conv2D(64, (3, 3)))
        BatchNormalization(axis=-1)
        classifier.add(Activation('relu'))
        classifier.add(Conv2D(128, (3, 3)))
        classifier.add(Activation('relu'))
        classifier.add(MaxPool2D(pool_size=(2, 2)))
        classifier.add(Conv2D(128, (3, 3)))
        classifier.add(Activation('relu'))
        classifier.add(MaxPool2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        BatchNormalization()
        classifier.add(Dense(512))
        BatchNormalization()
        classifier.add(Activation('relu'))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(6))
        classifier.add(Activation('softmax'))
        classifier.compile(optimizer='adam',
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

   #     rospy.sleep(1.0)

        self.network.load_weights('/home/alvarado/catkin_emiliano/src/assigment_publisher_subscriber/src/net2.h5')

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

	test_images = test_images[0:28, 28:56]
        
############
	imagenes_NN = test_images.reshape((1,28*28))
        imagenes_NN = imagenes_NN.astype('float32') / 255

        
        resultado = self.network.predict_classes(imagenes_NN)
 ###########
        print("Resultado: {}".format(resultado))
        
        velocidad = NormalizedSpeedCommand()
        
        if resultado == 5:
           velocidad.value = 0.1
        self.vel_pub.publish(velocidad)
        if resultado == 4:
           velocidad.value = 0.0
        self.vel_pub.publish(velocidad)
        if resultado == 0:
          velocidad.value = 0.3
        self.vel_pub.publish(velocidad)



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
