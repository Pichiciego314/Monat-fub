#!/usr/bin/env python
import rospy
import std_msgs.msg
from autominy_msgs.msg import SteeringAngle, NormalizedSpeedCommand, NormalizedSteeringCommand
from std_msgs.msg import String, Float32

def backwards_right(data, date):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.value)
    pub = rospy.Publisher('/control/command/normalized_wanted_speed', NormalizedSpeedCommand, queue_size=10)
    steering = rospy.Publisher('/control/command/normalized_wanted_steering', NormalizedSteeringCommand, queue_size=10)
    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()
    mensaje = NormalizedSpeedCommand()
    mensaje.header = h
    mensaje.value  = data.value

    j = std_msgs.msg.Header()
    j.stamp = rospy.Time.now()
    mensaje2 = NormalizedSteeringCommand()
    mensaje2.header = j
    mensaje2.value = date.value

    if date.value > 0:
       mensaje.vlaue = date.value * -1
    pub.publish(mensaje)



def comienzo():
    rospy.init_node('parking2', anonymous=True)
    pub = rospy.Publisher('/control/command/normalized_wanted_speed', NormalizedSpeedCommand, queue_size=10)
    steering = rospy.Publisher('/control/command/normalized_wanted_steering', NormalizedSteeringCommand, queue_size=10)

    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()
    mensaje = mensaje2.value()
    mensaje.header = h
    mensaje.value  = date.value

    j = std_msgs.msg.Header()
    j.stamp = rospy.Time.now()
    mensaje2 = NormalizedSteeringCommand()
    mensaje2.header = j
    mensaje2.value = 0.5
if __name__ == '__main__':
    comienzo()

