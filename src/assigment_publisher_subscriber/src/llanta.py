#!/usr/bin/env python
import rospy
import std_msgs.msg
from autominy_msgs.msg import SteeringAngle, NormalizedSpeedCommand
from std_msgs.msg import String


def callback(data):
    pub = rospy.Publisher('/control/command/normalized_wanted_speed', NormalizedSpeedCommand, queue_size=10)
    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()
    mensaje = NormalizedSpeedCommand()
    mensaje.header = h
    mensaje.value  = data.value
    
    if data.value < 0:
	mensaje.value = data.value * -1
    if data.value > 0:
        mensaje.value = data.value * -1
    
    pub.publish(mensaje)


def initial():
    rospy.init_node('llanta', anonymous=True)
    dumy = callback()
    
