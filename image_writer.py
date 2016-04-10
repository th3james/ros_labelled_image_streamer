"""Streams labelled image data to disk when turtlebot is moved
"""
__author__ =  'James Cox-Morton <th3james@fastmail.fm>'
__version__=  '0.1'

import sys, time

import numpy as np

import cv2
from cv_bridge import CvBridge

import roslib
import rospy

from sensor_msgs.msg import Image

VERBOSE=True
INPUT_TOPIC="/camera/rgb/image_raw/"
OUTPUT_DIR="/media/sdmount/sdcard/images/"

class image_feature:

  def __init__(self):
    rospy.init_node('image_feature')
      
    self.subscriber = rospy.Subscriber(INPUT_TOPIC,
        Image, self.callback,  queue_size = 1)
    
    self.last = rospy.Time.now()

  def callback(self, ros_data):
    '''Callback function of subscribed topic.
    Streams images to disk when button is depressed'''
    if rospy.Time.now() > self.last + rospy.Duration(1.0):
      self.last = rospy.Time.now()

      if VERBOSE :
        print 'received image of size: {0}*{1}'.format(ros_data.height, ros_data.width)

      cv_image = CvBridge().imgmsg_to_cv2(ros_data, desired_encoding="passthrough")
      with open(OUTPUT_DIR + 'hat.jpg') as file:
          cv2.imwrite(file.name, cv_image)

def main(args):
  '''Initializes and cleanup ros node'''
  ic = image_feature()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print "Shutting down ROS Image feature detector module"

if __name__ == '__main__':
    main(sys.argv)
