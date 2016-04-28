import tensorflow as tf
import cv2

import rect_training_data as training_data

RECORDS_TO_TRAIN_ON = 450

x = tf.placeholder(tf.float32, [None, training_data.IMAGE_POINTS])
y_ = tf.placeholder(tf.float32, [None, training_data.OUTPUT_CLASSES])

data = training_data.load_training_sets(RECORDS_TO_TRAIN_ON)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 3, 10])
b_conv1 = bias_variable([10])

x_image = tf.reshape(x, [-1]+training_data.IMAGE_DIMENSIONS)

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 10, 20])
b_conv2 = bias_variable([20])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([3 * 5 * 20, 360])
b_fc1 = bias_variable([360])
h_pool2_flat = tf.reshape(h_pool2, [-1, 3*5*20])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([360, training_data.OUTPUT_CLASSES])
b_fc2 = bias_variable([training_data.OUTPUT_CLASSES])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * y_conv))
optimiser = tf.train.AdamOptimizer(1e-4)
#train_step = optimiser.minimize(cross_entropy)
gradients = optimiser.compute_gradients(cross_entropy)
apply_gradients = optimiser.apply_gradients(gradients)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  xs, ys = data['train']
  batchSize = 100

  for i in xrange(0, len(xs), batchSize):
    batch_xs = xs[i:i+batchSize]
    batch_ys = ys[i:i+batchSize]

    print "train {0}-{1}".format(i, i+batchSize)

    #sess.run(train_step, feed_dict={
    #  x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    print("Cross entropy:")
    print(sess.run(cross_entropy, feed_dict={
      x: batch_xs, y_: batch_ys, keep_prob: 0.5}))

    print("First Gradient")
    print(sess.run(gradients[0], feed_dict={
      x: batch_xs, y_: batch_ys, keep_prob: 0.5})[0][0][0][0])

    print("Appling gradients")
    sess.run(apply_gradients, feed_dict={
      x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    print("Simple accuracy:")
    print(sess.run(accuracy, feed_dict={
        x: data['test'][0], y_: data['test'][1], keep_prob: 1}))

  weights = sess.run(tf.transpose(W_conv1))

  for weight in weights:
    img = weight.reshape(training_data.IMAGE_DIMENSIONS)
    cv2.imshow('image',img)
    cv2.waitKey(0)
