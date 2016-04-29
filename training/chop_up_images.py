import glob
import cv2
import numpy as np
import os

import training_data

TRAINING_DATA_PATH = "/Users/th3james/src/curricular/ros_labelled_image_streamer/training/training-data/rects/"
RECT_SIZE = {
  'width': 80,
  'height': 40,
  'x-step': 80,
  'y-step': 40,
}

data = training_data.load_data(200)

y_steps = range(
  0, training_data.IMAGE_DIMENSIONS[0], RECT_SIZE['y-step']
)
x_steps = range(
  0, training_data.IMAGE_DIMENSIONS[1], RECT_SIZE['x-step']
)

for i, image_data in enumerate(data[0]):
  image_folder = "{0}/{1}".format(TRAINING_DATA_PATH, i)
  if not os.path.exists(image_folder):
    os.makedirs(image_folder)
  image_data = image_data.reshape(training_data.IMAGE_DIMENSIONS)
  for y in y_steps:
    for x in x_steps:
      cv2.imwrite(
        "{0}/empty-{1}-{2}-{3}.jpg".format(image_folder, i, x, y),
        image_data[y:y+RECT_SIZE['height'], x:x+RECT_SIZE['width']]
      )
