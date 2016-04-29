import glob
import cv2
import numpy as np

import rect_training_data as training_data

PRESENT_IMAGE_DIR = "/Users/th3james/src/curricular/ros_labelled_image_streamer/training/training-data/rects/present/"

RECT_SIZE = {
  'width': 80,
  'height': 40
}

image_names = glob.glob(PRESENT_IMAGE_DIR + '*jpg')

for i, image_name in enumerate(image_names):
  image_data = cv2.imread(image_name)
  flipped_data = np.fliplr(image_data)

  image_name_no_prefix = image_name.split('.jpg', 0)[0]

  cv2.imwrite(
    "{0}-inverted.jpg".format(image_name_no_prefix), flipped_data
  )
