#!/usr/bin/env python

import rospy
import std_msgs.msg
from autominy_msgs.msg import SteeringAngle, NormalizedSpeedCommand, SteeringCommand, SpeedCommand, NormalizedSteeringCommand, Speed
from std_msgs.msg import String, Header

current = 1.0

def callback(data):    

    rospy.loginfo(rospy.get_caller_id() + "I heard %f", data.value)
    h = Header()
    h.stamp = Rospy.Time.now()
    mensaje = NormalizedSteeringCommand()
    mensaje.header = h
    mensaje.value = data.value

    if data.value < 0:
	mensaje.value = data.value * -1
    if data.value > 0:
        mensaje.value = data.value * -1
    
    pub.publish(mensaje)
    
def timer():
    h = Header()
    h.stamp = rospy.Time.now()
    mensaje = NormalizedSteeringCommand()
    mensaje.header = h
    mensaje.value = -0.45
    pub.publish(mensaje)


    
def initialize():
    pub = rospy.Publisher('/control/command/normalized_wanted_speed', NormalizedSpeedCommand, queue_size=1)
    steering = rospy.Publisher('/control/command/normalized_wanted_steering', NormalizedSteeringCommand, queue_size=1)
    rospy.Subscriber("/carstate/steering", NormalizedSteeringCommand, queue_size=1)
    rospy.Subscriber("/carstate/speed", Speed, callback)
    rospy.Timer(5.0, timer)
    h = Header()
    h.stamp = rospy.Time.now()
    mensaje = NormalizedSpeedCommand()
    mensaje.header = h
    mensaje.value = -0.45
    pub.publish(mensaje)

if __name__ == '__main__':
      initialize()
