import glob
import random
import cv2
import numpy as np
import os

TRAINING_DATA_PATH = "/Users/th3james/src/curricular/ros_labelled_image_streamer/training/training-data/rects/"
IMAGE_DIMENSIONS = [40, 80, 3][::-1]
IMAGE_POINTS = reduce(lambda x,y: x*y, IMAGE_DIMENSIONS)
OUTPUT_CLASSES = 2

def load_data(amount):
  paths = [TRAINING_DATA_PATH + name for name in os.listdir(TRAINING_DATA_PATH)]
  #directories = [path for path in paths if os.path.isdir(path)]
  directories = [TRAINING_DATA_PATH + dir + '/' for dir in ["0", "4", "7", "14", "11", "25", "32", "41", "65", "80", "95", "120", "126", "present"]]
  image_names = [path for dir in directories for path in glob.glob(dir + "/*jpg")]
  random.shuffle(image_names)
  
  images_data = np.zeros(
    (amount, IMAGE_POINTS), dtype=np.uint8
  )
  labels = np.zeros(
    (amount, OUTPUT_CLASSES), dtype=np.uint8
  )

  for i in range(0, amount):
    images_data[i] = load_image(image_names[i])
    labels[i] = extract_label(image_names[i])

  return (images_data, labels)

def load_training_sets(amount):
  images_data, labels = load_data(amount)

  ranges = {
    'train': range(0, int(amount*0.8)),
    'test': range(int(amount*0.2), amount),
  }

  f = lambda r: [images_data[r], labels[r]]
  return {k: f(v) for k, v in ranges.iteritems()}

def load_image(path):
  img = cv2.imread(path)
  # Flip the axes (Keras convolution uses this: https://github.com/fchollet/keras/issues/315)
  img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
  return img.flatten().transpose()

ACTION_TO_LABEL = {
  'empty': [1, 0],
  'present': [0, 1],
}

def extract_label(path):
  filename = path.split('/')[-1]
  return ACTION_TO_LABEL.get(filename.split('-')[0])
