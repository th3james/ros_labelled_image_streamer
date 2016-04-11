"""Streams labelled image data to disk when turtlebot is moved
"""
__author__ =  'James Cox-Morton <th3james@fastmail.fm>'
__version__=  '0.1'

import sys, time

import numpy as np
import readchar

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

class ParseCommands:
  KEY_MAPPINGS = {
    'w': "Forward",
    'a': "Left",
    'd': "Right",
  }

  @classmethod
  def get_current_command(cls):
    return cls.KEY_MAPPINGS.get(readchar.readkey(), None)

class TurtlebotMover:
  def __init__(self):
    self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=1)
    
    rospy.on_shutdown(self.shutdown)
  
  def go_forward(self):
    move_cmd = Twist()
    move_cmd.linear.x = 0.2 # 0.2 m/s
    move_cmd.angular.z = 0 # 0 radians/s

    self.cmd_vel.publish(move_cmd)

  def shutdown(self):
    # stop turtlebot
    rospy.loginfo("Stop TurtleBot")
    # a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
    self.cmd_vel.publish(Twist())
    # sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
    rospy.sleep(1)

class ImageTraining:

  def __init__(self, turtlebot_mover):
    self.turtlebot_mover = turtlebot_mover

    self.subscriber = rospy.Subscriber(INPUT_TOPIC,
        Image, self.callback,  queue_size = 1)
    
    self.last = rospy.Time.now()

  def callback(self, ros_data):
    '''Callback function of subscribed topic.
    Streams images to disk when button is depressed'''
    if rospy.Time.now() > self.last + rospy.Duration(SECONDS_BETWEEN_IMAGES):
      self.last = rospy.Time.now()
      
      command = ParseCommands.get_current_command()

      if command is not None:
        if VERBOSE:
          print 'received {0}*{1} image and command {2}'.format(ros_data.height, ros_data.width, command)

        cv_image = CvBridge().imgmsg_to_cv2(ros_data, desired_encoding="passthrough")
        filename = '{0}/{1}-{2}.jpg'.format(OUTPUT_DIR, command, str(self.last.secs))
        with open(filename, 'w+') as file:
            cv2.imwrite(file.name, cv_image)

        self.turtlebot_mover.go_forward()

def main(args):
  '''Initializes and cleanup ros node'''
  rospy.init_node('image_trainer')
  turtlebot_mover = TurtlebotMover()
  image_training = ImageTraining(turtlebot_mover)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print "Shutting down ROS Image feature detector module"

if __name__ == '__main__':
    main(sys.argv)
