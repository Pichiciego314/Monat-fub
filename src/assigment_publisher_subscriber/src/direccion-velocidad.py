#!/usr/bin/env python
import rospy
import std_msgs.msg
from autominy_msgs.msg import SteeringAngle, NormalizedSpeedCommand
from std_msgs.msg import String


def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.value)
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



if __name__ == '__main__':
    def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
        rospy.init_node('direccion-velocidad', anonymous=True)
        dummy = callback()
        rospy.Rate(10)

        rospy.Subscriber("/carstate/steering", SteeringAngle, callback)


    
