#!/usr/bin/env python
import numpy as np
import rospy
import tf
import rospkg
from std_msgs.msg import Float64, Header
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
import sys
import matplotlib.pyplot as plt
from scipy import stats
from std_msgs.msg import Int16, UInt8, UInt16
from time import localtime, strftime

speed_value =150 #speed value
speed = +speed_value # initial direction is backward
steering_angle = 0 


max_y = 2.0 # initial y limitation is 2 meter

inlier_dist = 0.05  # max distance for inliers (in meters) since walls are very flat this can be low
drive_duration_max = 5  # number of seconds to drive
plotting = True  # whether to plot output

manual_mode = False  # in manual mode we don't actually send commands to the motor

mask_angles = True  # whether to mask any angle that's not on the right side of the car

sample_count = 50  # number RANSAC samples to take

turn_radii = []  # store the detected turn radii in this list
radius_theta = [] # store the detected turn radii in this list vs theta
servo_feedback =[]
wall_angle = 0
target_angle = wall_angle #mask the lidar points
add_pi = np.pi
last_theta = 0

pub_stop_start = rospy.Publisher("/control/command/normalized_wanted_speed", Int16, queue_size=100, latch=True)
pub_speed = rospy.Publisher("/manual_control/speed", Int16, queue_size=100, latch=True)
pub_steering = rospy.Publisher("/steering", UInt8, queue_size=100, latch=True)
steering_angle_feedback=0



 
        

def main(args):
    global steering_angle
    rospy.init_node("angle_calibration")
    if len(args) > 1:
	try:
    	    steering_angle = int(args[1])
            rospy.Subscriber("/sensor/rplidar/scan", LaserScan, scan_callback, queue_size=1)
	    rospy.Subscriber("/control/command/normalized_wanted_steering", NormalizedSteeringCommand, steering_feedback_callback, queue_size=1)  # small queue for only reading recent data

        except rospy.ROSInterruptException:
	       pass
	else:
	     print("please provide a steering setting from [0,180]") 

	if plotting:
	   plt.show()  # block until plots are closed
 	else:
            rospy.spin()
   
if __name__ == '__main':
    main(sys.argv)
