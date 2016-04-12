import glob
import random
import cv2
import numpy as np

TRAINING_DATA_PATH = "/Users/th3james/src/curricular/ros_labelled_image_streamer/training/training-data/"
IMAGE_POINTS = 921600
OUTPUT_CLASSES = 3
RECORDS_TO_LOAD = 10

def load_data():
  print(TRAINING_DATA_PATH + '*jpg')
  image_names = glob.glob(TRAINING_DATA_PATH + '*jpg')
  random.shuffle(image_names)
  
  images_data = np.zeros(
    (RECORDS_TO_LOAD, IMAGE_POINTS), dtype=np.uint8
  )
  labels = np.zeros(
    (RECORDS_TO_LOAD, OUTPUT_CLASSES), dtype=np.uint8
  )

  for i in range(0, RECORDS_TO_LOAD):
    images_data[i] = load_image(image_names[i])
    labels[i] = extract_label(image_names[i])

  ranges = {
    'train': range(0, int(RECORDS_TO_LOAD*0.6)),
    'test': range(int(RECORDS_TO_LOAD*0.6), RECORDS_TO_LOAD),
  }

  f = lambda r: [images_data[r], labels[r]]
  return {k: f(v) for k, v in ranges.iteritems()}

def load_image(path):
  img = cv2.imread(path)
  return img.flatten().transpose()

ACTION_TO_LABEL = {
  'Left': [1, 0, 0],
  'Forward': [0, 1, 0],
  'Right': [0, 0, 1],
}

def extract_label(path):
  filename = path.split('/')[-1]
  return ACTION_TO_LABEL.get(filename.split('-')[0])
