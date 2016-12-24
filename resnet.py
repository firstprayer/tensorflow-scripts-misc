from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import time
import os

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

FLAGS = None
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
LABEL_NUM = 10

def load_data():
  mnist_data = input_data.read_data_sets(FLAGS.data_dir)
  return mnist_data


def weight_variable(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name='W')

def bias_variable(shape):
  return tf.Variable(tf.constant(0.1, shape=shape), name='b')


class CNN:

  def __init__(self, session, X_placeholder, Y_placeholder, y_pred, loss):
    self.session = session
    self.X_placeholder = X_placeholder
    self.Y_placeholder = Y_placeholder
    self.Y_pred = y_pred
    self.loss = loss
    
    # Additional steps.
    self.train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y_placeholder, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  def init(self):
    self.session.run(tf.global_variables_initializer())

  def train(self, X, Y):
    self.train_step.run(feed_dict={self.X_placeholder: X, self.Y_placeholder: Y})

  def test(self, X, Y):
    return self.accuracy.eval(feed_dict={self.X_placeholder: X, self.Y_placeholder: Y})


def generate_lenet_graph(inputX):
  with tf.device('/cpu:0'):
    num_fm_conv1 = 32
    with tf.name_scope('conv1'):
      W_conv1 = weight_variable([5, 5, 1, num_fm_conv1])
      b_conv1 = bias_variable([num_fm_conv1])
      conv1 = tf.nn.conv2d(inputX, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
      h1 = tf.nn.relu(conv1 + b_conv1)
      pool1 = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    # Second layer of convolution
    with tf.name_scope('conv2'):
      num_fm_conv2 = 64
      W_conv2 = weight_variable([5, 5, num_fm_conv1, num_fm_conv2])
      b_conv2 = bias_variable([num_fm_conv2])
      conv2 = tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
      h2 = tf.nn.relu(conv2 + b_conv2)
      pool2 = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    # FC layer
    with tf.name_scope('fc'):
      output_size = 7 * 7 * num_fm_conv2
      fc_output_size = 1024
      pool2_reshape = tf.reshape(pool2, [-1, output_size])
      W_fc = weight_variable([output_size, fc_output_size])
      b_fc = weight_variable([fc_output_size])
      h_fc = tf.nn.relu(tf.matmul(pool2_reshape, W_fc) + b_fc)
    
    # Output LR.
    with tf.name_scope('pred'):
      W_pred = weight_variable([fc_output_size, LABEL_NUM])
      b_pred = bias_variable([LABEL_NUM])
      Y_pred_activation = tf.matmul(h_fc, W_pred) + b_pred
      
    return Y_pred_activation

def build_train_net(session, X_placeholder, Y_placeholder):
  # First layer of convolution.
  image = tf.reshape(X_placeholder, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
  Y_pred_activation = generate_lenet_graph(image)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y_pred_activation, Y_placeholder))
  return CNN(session, X_placeholder, Y_placeholder, Y_pred_activation, loss)

def simple_lr(X_placeholder, Y_placeholder):
  N = IMAGE_WIDTH * IMAGE_HEIGHT
  W = weight_variable([N, LABEL_NUM])
  b = bias_variable([LABEL_NUM])

  h = tf.matmul(X_placeholder, W) + b
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(h, Y_placeholder))
  return CNN(session, X_placeholder, Y_placeholder, h, loss)


def main():
  mnist_data = load_data()
  batch_size = 32

  X_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name='X')
  Y_placeholder = tf.placeholder(tf.int32, shape=[None, LABEL_NUM], name='Y')
  
  session = tf.InteractiveSession()

  net = build_train_net(session, X_placeholder, Y_placeholder)
  net.init()

  saver = tf.train.Saver()
  meta_graph_def = tf.train.export_meta_graph(
      filename=os.path.join(FLAGS.save_dir, 'model.meta'), collection_list=['variables'])
  for i in range(10000):
    start_time = time.time()
    X, Y = mnist_data.train.next_batch(batch_size)
    Y = np.eye(LABEL_NUM)[Y]
    net.train(X, Y)
    duration = time.time() - start_time

    if i % 1000 == 999:
      train_accuracy = net.test(X, Y)
      print('train_accuracy: %f (walltime: %.3f)' % (train_accuracy, duration))
      saver.save(session, os.path.join(FLAGS.save_dir, 'model.ckpt'), global_step=i)

  test_Y = np.eye(LABEL_NUM)[mnist_data.test.labels]
  test_accuracy = net.test(mnist_data.test.images, test_Y)
  print('test_accuracy: %f' % test_accuracy)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='mnist_data',
      help='data path',
  )
  parser.add_argument(
      '--save_dir',
      type=str,
      default='mnist_model',
      help='save path',
  )
  FLAGS, _ = parser.parse_known_args()
  main()
