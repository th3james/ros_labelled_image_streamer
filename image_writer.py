"""Streams labelled image data to disk when turtlebot is moved
"""
__author__ =  'James Cox-Morton <th3james@fastmail.fm>'
__version__=  '0.1'

import sys, time

import numpy as np
import curses

import cv2
from cv_bridge import CvBridge

import roslib
import rospy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

VERBOSE=True
INPUT_TOPIC="/camera/rgb/image_raw/"
OUTPUT_DIR="/media/sdmount/sdcard/images/"
SECONDS_BETWEEN_IMAGES=1.0


def get_current_command(stdscr):
  mappings = {
    119: "Forward",
    97: "Left",
    100: "Right",
  }
  curses.flushinp()
  ch = stdscr.getch()
  return mappings.get(ch, None)

class image_feature:

  def __init__(self, stdscr):
    self.stdscr = stdscr
    rospy.init_node('image_feature')
      
    self.subscriber = rospy.Subscriber(INPUT_TOPIC,
        Image, self.callback,  queue_size = 1)
    
    self.last = rospy.Time.now()

  def callback(self, ros_data):
    '''Callback function of subscribed topic.
    Streams images to disk when button is depressed'''
    if rospy.Time.now() > self.last + rospy.Duration(SECONDS_BETWEEN_IMAGES):
      self.last = rospy.Time.now()
      
      command = get_current_command(self.stdscr)

      if command is not None:
        if VERBOSE :
          print 'received image of size: {0}*{1}'.format(ros_data.height, ros_data.width)

        cv_image = CvBridge().imgmsg_to_cv2(ros_data, desired_encoding="passthrough")
        filename = OUTPUT_DIR + command + str(self.last.secs) + '.jpg'
        self.stdscr.addstr(filename)
        with open(filename, 'w+') as file:
            cv2.imwrite(file.name, cv_image)


def curses_main(stdscr):
  ic = image_feature(stdscr)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print "Shutting down ROS Image feature detector module"

def main(args):
  '''Initializes and cleanup ros node'''
  curses.wrapper(curses_main)

if __name__ == '__main__':
    main(sys.argv)
