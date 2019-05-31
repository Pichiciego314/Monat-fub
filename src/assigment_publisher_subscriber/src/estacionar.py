#!/usr/bin/env python

import numpy as np
import rospy
import tf
import rospkg
from autominy_msgs.msg import  SteeringAngle, NormalizedSpeedCommand, NormalizedSteeringCommand
from std_msgs.msg import Float64, Header
from nav_msgs.msg import Odometry


class parking:
    def __init__(self):
        self.vel_pub = rospy.Publisher("/control/command/normalized_wanted_speed", NormalizedSpeedCommand, queue_size=0)
        self.dir_pub = rospy.Publisher("/control/command/normalized_wanted_steering", NormalizedSteeringCommand, queue_size=0)

    def simple_parking(self):

        # Definiendo mensaje de velocidad
        velocidad = NormalizedSpeedCommand()


        # Definiendo mensaje de direccion
        direccion = NormalizedSteeringCommand()

        rospy.sleep(1.0)

        ''' Aqui va la magia de seleccionar que 
        direccion y vel se desea (velocidad.value = ?, direccion.value = ?) '''
#here are the values assigned to speed amd steering
        # Primeros valores
        velocidad.value = -0.2
        direccion.value = 0.7
        self.vel_pub.publish(velocidad)
        self.dir_pub.publish(direccion)
        # Parar tiempo
        rospy.sleep(2.0)
        # Segundos valores
        velocidad.value = -0.2
        direccion.value = -0.7
        #self.vel_pub.publish(velocidad)
        self.dir_pub.publish(direccion)
        # Parar tiempo
        rospy.sleep(1.5)
        #terceros valores
        velocidad.value = 0.2
        direccion.value = 0.2
        self.vel_pub.publish(velocidad)
        self.dir_pub.publish(direccion)
        rospy.sleep(1.0)
        

#para parar
        velocidad.value = 0.0
        direccion.value = 0.0
        self.vel_pub.publish(velocidad)
        self.dir_pub.publish(direccion)
        rospy.sleep(10.0)


if __name__ == '__main__':
    try:
        rospy.init_node('estacionar')
        hola = parking() # Crea un objeto para ejecutar parking
        rate = rospy.Rate(10) # Que tan rapido quiero que se ejecute el programa
        while not rospy.is_shutdown():
            hola.simple_parking()
            rate.sleep()
    except rospy.ROSInterruptException:
        print("Algo raro esta pasando")
        pass
