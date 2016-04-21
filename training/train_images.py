import tensorflow as tf
import cv2

import image_training_data as training_data

RECORDS_TO_TRAIN_ON = 550

x = tf.placeholder(tf.float32, [None, training_data.IMAGE_POINTS])
W = tf.Variable(tf.zeros([training_data.IMAGE_POINTS, training_data.OUTPUT_CLASSES]))
b = tf.Variable(tf.zeros([training_data.OUTPUT_CLASSES]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, training_data.OUTPUT_CLASSES])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
data = training_data.load_training_sets(RECORDS_TO_TRAIN_ON)

#with tf.Session() as sess:
#  sess.run(init)
#  batch_xs, batch_ys = data['train']
#  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#
#  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#  print(sess.run(accuracy, feed_dict={x: data['test'][0], y_: data['test'][1]}))
#
#  weights = sess.run(tf.transpose(W))
#
#  for weight in weights:
#    img = weight.reshape(training_data.IMAGE_DIMENSIONS)
#    cv2.imshow('image',img)
#    cv2.waitKey(0)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1]+training_data.IMAGE_DIMENSIONS)

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, training_data.OUTPUT_CLASSES])
b_fc2 = bias_variable([training_data.OUTPUT_CLASSES])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  
  batch_xs, batch_ys = data['train']
  train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
  
  print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
