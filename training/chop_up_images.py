import glob
import cv2
import numpy as np

import training_data

RECT_SIZE = {
  'height': 20,
  'width': 40,
}

data = training_data.load_data(1)

image_data = data[0]
image_data = image_data.reshape(training_data.IMAGE_DIMENSIONS)

y_steps = range(
  0, training_data.IMAGE_DIMENSIONS[0], RECT_SIZE['height']/2
)
x_steps = range(
  0, training_data.IMAGE_DIMENSIONS[1], RECT_SIZE['width']/2
)

for y in y_steps:
  for x in x_steps:
    cv2.imshow('img',
      image_data[y:y+RECT_SIZE['height'], x:x+RECT_SIZE['width']]
    )
