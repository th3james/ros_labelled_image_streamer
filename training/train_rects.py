from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
import cv2

import rect_training_data as training_data

RECORDS_TO_TRAIN_ON = 1000
batchSize = 50

data = training_data.load_training_sets(RECORDS_TO_TRAIN_ON)

model = Sequential()

model.add(Convolution2D(10, 3, 3, border_mode='same', input_shape=training_data.IMAGE_DIMENSIONS, bias=True))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Convolution2D(3, 3, 3, border_mode='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())
#model.add(Dense(output_dim=1024))
#model.add(Activation("relu"))

model.add(Dense(output_dim=100))
model.add(Activation("relu"))

model.add(Dense(output_dim=50))
model.add(Activation("relu"))

model.add(Dense(output_dim=10))
model.add(Activation("relu"))

#model.add(Dropout(0.9))
model.add(Dropout(1))

model.add(Dense(output_dim=2, bias=True))
model.add(Activation("softmax"))

optimiser = Adam(lr=0.00005)
model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=['accuracy'])

print model.summary()

# Do a training
xs, ys = data['train']
test_x, test_y = data['test']

#for i in xrange(0, 4):
#    print "image is: {}".format(test_y[i])
#    img = test_x[i].reshape(training_data.IMAGE_DIMENSIONS)
#    cv2.imshow('image',img)
#    cv2.waitKey(0)

def reshape_x(x, img_dimensions):
  return x.reshape([x.shape[0]]+img_dimensions)

runNo = 0
for i in xrange(0, len(xs), batchSize):
  batch_xs = xs[i:i+batchSize]
  batch_ys = ys[i:i+batchSize]

  print "train {0}-{1}".format(i, i+batchSize)

  model.train_on_batch(reshape_x(batch_xs, training_data.IMAGE_DIMENSIONS), batch_ys)

  loss_and_metrics = model.evaluate(reshape_x(test_x, training_data.IMAGE_DIMENSIONS), test_y, batch_size=32)
  print "{}: {}".format(model.metrics_names[1], loss_and_metrics[1])
