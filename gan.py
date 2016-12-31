from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import time
import os
import cPickle as pickle

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
from visualize_helper import plot_images

FLAGS = None

def load_data():
  mnist_data = input_data.read_data_sets(FLAGS.data_dir)
  return mnist_data


class Layer:

  def __init__(self, prefix=None):
    self.prefix = prefix

  def get_name(self, name):
    if self.prefix is not None:
      return self.prefix + '_' + name
    return name

  def linear_layer(self, X, hidden_unit_num):
    W = tf.get_variable(
        self.get_name("weights"), \
        [X.get_shape().as_list()[-1], hidden_unit_num], \
        initializer=tf.random_normal_initializer())
    b = tf.get_variable(
        self.get_name("bias"), \
        [hidden_unit_num], \
        initializer=tf.constant_initializer(0.0))
    return tf.matmul(X, W) + b


  def relu_hidden_layer(self, input_vec, hidden_unit_num):
    return tf.nn.relu(self.linear_layer(input_vec, hidden_unit_num))


  def sigmoid_hidden_layer(self, input_vec, hidden_unit_num):
    return tf.nn.sigmoid(self.linear_layer(input_vec, hidden_unit_num))


  def upsampling_2d(self, image_nhwc, height_multiplier, width_multiplier):
    '''
    Assume the image tensor is in BatchSize / Channel / Height / Width order
    Copied implementation from Keras. https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py
    '''
    old_shape = image_nhwc.get_shape().as_list()
    old_shape_hw = old_shape[1: 3]
    new_shape_hw = old_shape_hw * tf.constant(np.array([height_multiplier, width_multiplier]).astype('int32'))
    new_image_nhwc = tf.image.resize_nearest_neighbor(image_nhwc, new_shape_hw)
    new_shape = (None, old_shape_hw[0] * height_multiplier,
        old_shape_hw[1] * width_multiplier, old_shape[-1])
    new_image_nhwc.set_shape(new_shape)
    return new_image_nhwc


  def convolutional_2d(self, image_nhwc, feature_map_num, filter_height, filter_width):
    W = tf.get_variable(
        self.get_name("weights"), \
        [filter_height, filter_width, image_nhwc.get_shape().as_list()[-1], feature_map_num], \
        initializer=tf.random_normal_initializer())
    b = tf.get_variable(
        self.get_name("bias"),
        [feature_map_num], \
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(image_nhwc, W, strides=[1, 1, 1, 1], padding='SAME')
    return conv

  def flatten(self, X):
    dims = X.get_shape().as_list()[1: ]
    print(dims)
    new_dim = reduce(lambda x, y : x * y, dims, 1)
    return tf.reshape(X, shape=(-1, new_dim))


def MaxPooling2D(X, pool_height, pool_width):
  return tf.nn.max_pool(X, ksize=[1, pool_height, pool_width, 1], \
      strides=[1, pool_height, pool_width, 1], padding='SAME')


PRIOR_DIM  = 100
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNEL = 1
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL

def sample_from_prior():
  return np.random.uniform(-1, 1, (FLAGS.batch_size, PRIOR_DIM))

def GenerativeNet(prior):
  h1 = Layer('h1').relu_hidden_layer(prior, 2000)
  h2 = Layer('h2').relu_hidden_layer(h1, 1000)
  X_gen = Layer('X_gen').sigmoid_hidden_layer(h2, IMAGE_SIZE)
  return X_gen

def GenerativeConvolutionNet(prior):
  # Following https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py
  # model.add(Dense(input_dim=100, output_dim=1024))
  fc1 = Layer('fc').linear_layer(prior, 1024)       
  # model.add(Activation('tanh'))
  fc1_tanh = tf.nn.tanh(fc1)  
  # model.add(Dense(128*7*7))
  fc2 = Layer('fc2').linear_layer(fc1_tanh, 128 * 7 * 7)  
  # model.add(BatchNormalization())
  bn = fc2              # lack of good implementation with BatchNormalization in tensorflow tf.nn.batch_normalization(fc2)  
  # model.add(Activation('tanh'))
  bn_tanh = tf.nn.tanh(bn)  
  # model.add(Reshape((128, 7, 7), input_shape=(128*7*7,)))
  reshape_img1 = tf.reshape(bn_tanh, shape=(-1, 7, 7, 128))  
  # model.add(UpSampling2D(size=(2, 2)))
  upsample_img1 = Layer('upsample_img1').upsampling_2d(reshape_img1, 2, 2)  
  # model.add(Convolution2D(64, 5, 5, border_mode='same'))
  conv1 = Layer('conv1').convolutional_2d(upsample_img1, 64, 5, 5)
  # model.add(Activation('tanh'))
  conv1_tanh = tf.nn.tanh(conv1)
  # model.add(UpSampling2D(size=(2, 2)))
  upsample_img2 = Layer('upsample_img2').upsampling_2d(conv1_tanh, 2, 2)
  # model.add(Convolution2D(1, 5, 5, border_mode='same'))
  conv2 = Layer('conv2').convolutional_2d(upsample_img2, 1, 5, 5)
  # model.add(Activation('tanh'))
  conv2_tanh = tf.nn.tanh(conv2)
  print(conv2_tanh.get_shape().as_list())
  return conv2_tanh

def DiscriminativeNet(X):
  flat = tf.reshape(X, shape=(-1, IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL))
  h1 = Layer('h1').relu_hidden_layer(flat, 1000)
  h2 = Layer('h2').relu_hidden_layer(h1, 400)
  return Layer('Y_pred').linear_layer(h2, 1)

def DiscriminativeConvolutionNet(X):
  # model.add(Convolution2D(64, 5, 5, border_mode='same', input_shape=(1, 28, 28)))
  conv1 = Layer('conv1').convolutional_2d(X, 64, 5, 5)
  # model.add(Activation('tanh'))
  conv1_tanh = tf.nn.tanh(conv1)
  # model.add(MaxPooling2D(pool_size=(2, 2)))
  pool1 = MaxPooling2D(conv1_tanh, 2, 2)
  # model.add(Convolution2D(128, 5, 5))
  conv2 = Layer('conv2').convolutional_2d(pool1, 128, 5, 5)
  # model.add(Activation('tanh'))
  conv2_tanh = tf.nn.tanh(conv2)
  # model.add(MaxPooling2D(pool_size=(2, 2)))
  pool2 = MaxPooling2D(conv2_tanh, 2, 2)
  # model.add(Flatten())
  flat = Layer('flat').flatten(pool2)
  print(conv1.get_shape().as_list(),
        pool1.get_shape().as_list(),
        conv2.get_shape().as_list(),
        pool2.get_shape().as_list(),
        flat.get_shape().as_list())
  # model.add(Dense(1024))
  fc = Layer('fc').linear_layer(flat, 1024)
  # model.add(Activation('tanh'))
  fc_tanh = tf.nn.tanh(fc)
  # model.add(Dense(1))
  # model.add(Activation('sigmoid'))
  return Layer('Y_pred').linear_layer(fc_tanh, 1)
  

def main():
  mnist_data = load_data()
  images, labels = mnist_data.train.images, mnist_data.train.labels
  label_to_images, label_to_pointer = {}, {}
  for l in range(10):
    indices = np.where(labels == l)
    label_to_images[l] = images[indices, :].squeeze()
    label_to_pointer[l] = 0
  def next_image_batch_for_label(label):
    p = label_to_pointer[label]
    imgs = label_to_images[label]
    
    # Can't find a full batch.
    if p > imgs.shape[0] - FLAGS.batch_size:
      p = 0
    new_p = min(p + FLAGS.batch_size, imgs.shape[0])
    label_to_pointer[label] = new_p
    batch = imgs[p: new_p, :].reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    return batch
   
  # Construct G.
  graph = tf.Graph()
  with graph.as_default():
    with tf.Session() as session:
      prior_placeholder = tf.placeholder(tf.float32, shape=[None, PRIOR_DIM], name='prior')
      with tf.variable_scope('Generative'):
        X_gen = GenerativeConvolutionNet(prior_placeholder)
        Y_gen_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name='Y_gen')

      # Construct D. With variable sharing.
      with tf.variable_scope('Discriminative'):
        X_placeholder = tf.placeholder(tf.float32,
                shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], name='X')
        Y_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name='Y')
        with tf.variable_scope('DiscriminativeShared') as share_scope:
          #Y_gen_pred = DiscriminativeNet(X_gen) # DiscriminativeConvolutionNet(X_gen)
          Y_gen_pred = DiscriminativeConvolutionNet(X_gen)
          share_scope.reuse_variables()
          # Y_pred = DiscriminativeNet(X_placeholder) # DiscriminativeConvolutionNet(X_placeholder)
          Y_pred = DiscriminativeConvolutionNet(X_placeholder)

      D_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(Y_gen_pred, Y_gen_placeholder)) + \
               tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(Y_pred, Y_placeholder))
      G_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(Y_gen_pred, Y_gen_placeholder))
      print([v.name for v in graph.get_collection('trainable_variables')])
      G_parameter_list = [v for v in graph.get_collection('trainable_variables')
                          if v.name.startswith('Generative/')]
      D_parameter_list = [v for v in graph.get_collection('trainable_variables') 
                          if v.name.startswith('Discriminative/')]
      print('G_parameter_list: ', [v.name for v in G_parameter_list])
      print('D_parameter_list: ', [v.name for v in D_parameter_list])
      
      epsilon = 1e-3
      G_train = tf.train.AdamOptimizer(epsilon).minimize(G_loss, var_list=G_parameter_list)
      D_train = tf.train.AdamOptimizer(epsilon).minimize(D_loss, var_list=D_parameter_list)
      
      session.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
      for iter_idx in range(20000):
        for _ in range(FLAGS.step_k):
          X_data = next_image_batch_for_label(0)  # Use 0 for testing for now.
          prior_sample = sample_from_prior()
          session.run(D_train, feed_dict={
            X_placeholder: X_data,
            Y_placeholder: np.asarray([[1.0] for _ in range(FLAGS.batch_size)]),  # 1 means True
            prior_placeholder: prior_sample,
            Y_gen_placeholder: np.asarray([[0.0] for _ in range(FLAGS.batch_size)]),
          })

        prior_sample = sample_from_prior()
        session.run(G_train, feed_dict={
          prior_placeholder: prior_sample,
          Y_gen_placeholder: np.asarray([[1.0] for _ in range(FLAGS.batch_size)]),
        })

        if iter_idx % 500 == 0 and iter_idx > 0:
          saver.save(session, os.path.join(FLAGS.save_dir, 'model.ckpt'), global_step=iter_idx)
          # Generate a couple images and save locally.
          prior_sample = sample_from_prior()
          X_gen_values = session.run(X_gen, feed_dict={prior_placeholder: prior_sample})
          images = X_gen_values.reshape((prior_sample.shape[0], 28, 28))
          plot_images(images, 8, FLAGS.batch_size // 8,
              save_filename='%s/X_Gen_%d.jpg' % (FLAGS.save_dir, iter_idx))
          print("Iter %d finished" % iter_idx)


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
      default='gan_model',
      help='save path',
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=32,
      help='Batch size for training',
  )
  parser.add_argument(
      '--step_k',
      type=int,
      default=1,
      help='Epoch for training D during 1 loop of G',
  ) 
  FLAGS, _ = parser.parse_known_args()
  main()
