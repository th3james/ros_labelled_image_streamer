import cv2
import glob
import random

TRAINING_DATA_PATH = "/Users/th3james/src/curricular/ros_labelled_image_streamer/training/training-data/"
example_name = "Forward-1460378468.jpg"

def load_data():
  print(TRAINING_DATA_PATH + '*jpg')
  image_names = glob.glob(TRAINING_DATA_PATH + '*jpg')
  random.shuffle(image_names)
  
  return {
    'train': map(load_image_and_label, image_names[0:2]),
    'cross': map(load_image_and_label, image_names[2:2]),
    'test': map(load_image_and_label, image_names[3:4]),
  }
  
def load_image_and_label(path):
  return [
    load_image(path), extract_label(path)
  ]

def load_image(path):
  img = cv2.imread(path)
  return img.flatten()

ACTION_TO_LABEL = {
  'Left': [1, 0, 0],
  'Forward': [0, 1, 0],
  'Right': [0, 0, 1],
}

def extract_label(path):
  filename = path.split('/')[-1]
  return ACTION_TO_LABEL.get(filename.split('-')[0])
