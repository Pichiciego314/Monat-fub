#!/usr/bin/env python
#here are impoting the libraries the program will be used
import numpy as np
import rospy
import tf
import rospkg
from autominy_msgs.msg import  SteeringAngle, NormalizedSpeedCommand, NormalizedSteeringCommand
from std_msgs.msg import Float64, Header
from nav_msgs.msg import Odometry


class parking:
    #Defiition of permanent variables
    def __init__(self):
        # The subscribers for the topic of the car
        self.vel_pub = rospy.Publisher("/control/command/normalized_wanted_speed", NormalizedSpeedCommand, queue_size=0)
        self.dir_pub = rospy.Publisher("/control/command/normalized_wanted_steering", NormalizedSteeringCommand, queue_size=0)

    def simple_parking(self):
        # here comes the values I will asign to the topics
        # definition of message for speed
        velocidad = NormalizedSpeedCommand()


        # definition of message for steering
        direccion = NormalizedSteeringCommand()

        rospy.sleep(1.0)
#here are the values assigned to speed amd steering
        # first values
        velocidad.value = -0.2
        direccion.value = 0.7
        self.vel_pub.publish(velocidad)
        self.dir_pub.publish(direccion)
        # use this values for this period of time
        rospy.sleep(2.0)
        # second values
        velocidad.value = -0.2
        direccion.value = -0.7
        self.vel_pub.publish(velocidad)
        self.dir_pub.publish(direccion)
        # use this values for this period of time
        rospy.sleep(1.5)
        #third values
        velocidad.value = 0.2
        direccion.value = 0.2
        self.vel_pub.publish(velocidad)
        self.dir_pub.publish(direccion)
        rospy.sleep(1.0)
        
        # for stop, in this program the stop was by using ctrl+c
        velocidad.value = 0.0
        direccion.value = 0.0
        self.vel_pub.publish(velocidad)
        self.dir_pub.publish(direccion)
        rospy.sleep(10.0)

# main function
if __name__ == '__main__':
    try:
        rospy.init_node('estacionar')
        hola = parking() # create an object to execute parking
        rate = rospy.Rate(10) # how fast the program will be executed
        while not rospy.is_shutdown():
            hola.simple_parking()
            rate.sleep()
    except rospy.ROSInterruptException:
        print("Algo raro esta pasando")
        pass
