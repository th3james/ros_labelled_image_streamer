import tensorflow as tf

import training_data

IMAGE_POINTS = 921600
OUTPUT_CLASSES = 3


x = tf.placeholder(tf.float32, [None, IMAGE_POINTS])
W = tf.Variable(tf.zeros([IMAGE_POINTS, OUTPUT_CLASSES]))
b = tf.Variable(tf.zeros([OUTPUT_CLASSES]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, OUTPUT_CLASSES])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
data = training_data.load_data()

with tf.Session() as sess:
  sess.run(init)
  batch_xs, batch_ys = data['train']
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  print(sess.run(accuracy, feed_dict={x: data['test'][0], y_: data['test'][1]}))


