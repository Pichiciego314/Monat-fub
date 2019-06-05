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



class neural:
    def __init__(self):
        self.net_image = []
        self.camara_sub = rospy.Subscriber("/sensors/camera/color/image_raw", Image, self.get_image)
        self.img_pub = rospy.Publisher("/imagen/nuevo_topico", Image, queue_size=10)
        self.bridge = CvBridge()
        self.vel_pub = rospy.Publisher("/control/command/normalized_wanted_speed", NormalizedSpeedCommand, queue_size=10)
      
        #convolutional neural network structure
        self.classifier = models.Sequential()
        self.classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
        BatchNormalization(axis=-1)  # Axis -1 is always the features axis
        self.classifier.add(Activation('relu'))
        self.classifier.add(Conv2D(64, (3, 3)))
        BatchNormalization(axis=-1)
        self.classifier.add(Activation('relu'))
        self.classifier.add(MaxPool2D(pool_size=(2, 2)))
        BatchNormalization(axis=-1)
        self.classifier.add(Conv2D(64, (3, 3)))
        BatchNormalization(axis=-1)
        self.classifier.add(Activation('relu'))
        self.classifier.add(Conv2D(128, (3, 3)))
        self.classifier.add(Activation('relu'))
        self.classifier.add(MaxPool2D(pool_size=(2, 2)))
        self.classifier.add(Conv2D(128, (3, 3)))
        self.classifier.add(Activation('relu'))
        self.classifier.add(MaxPool2D(pool_size=(2, 2)))
        self.classifier.add(Flatten())
        BatchNormalization()
        self.classifier.add(Dense(512))
        BatchNormalization()
        self.classifier.add(Activation('relu'))
        self.classifier.add(Dropout(0.2))
        self.classifier.add(Dense(6))
        self.classifier.add(Activation('softmax'))
        self.classifier.compile(optimizer='adam',
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
        #here are the uploadings of the weights
        self.classifier.load_weights('/home/alvarado/catkin_emiliano/src/assigment_publisher_subscriber/src/net4.h5')
        #here the image is being converted into black and white 
        gray=cv2.cvtColor(self.net_image, cv2.COLOR_BGR2GRAY)
        ret2,thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        test_images = thresh1
	print("Tamanio imagen: {}".format(test_images.shape))
	test_images = cv2.resize(test_images, (56,56), interpolation = cv2.INTER_AREA) 
        test_images = test_images[0:28, 28:56] #here image is being resizedinto something the computer can reads

        try:
            ros_img = self.bridge.cv2_to_imgmsg(test_images)
            self.img_pub.publish(ros_img)
        except CvBridgeError as e:
            print(e)

        
############
	imagenes_NN = test_images.reshape((1, 28, 28, 1))
        imagenes_NN = imagenes_NN.astype('float32') / 255

        
        resultado = self.classifier.predict_classes(imagenes_NN)
 ###########
        print("Resultado: {}".format(resultado))
        #return
        #here are set up the conditions for the numbers the net sees.
        velocidad = NormalizedSpeedCommand()
        
        if resultado == 5:
           velocidad.value = 0.5
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

        dummy = parking()#creates an obect to execute neural
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            dummy.simple_parking()
            rate.sleep()
    except rospy.ROSInterruptException:
        print("Algo raro esta pasando")
        pass		
