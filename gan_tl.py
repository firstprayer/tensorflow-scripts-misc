from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import tensorlayer as tl
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

def GenerativeConvolutionNet(prior, is_train, reuse):
  with tf.variable_scope("Generative", reuse=reuse):
    tl.layers.set_name_reuse(reuse)

    network = tl.layers.InputLayer(prior, name='g/prior')
    # Following https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py
    # model.add(Dense(input_dim=100, output_dim=1024))
    # model.add(Activation('tanh'))
    network = tl.layers.DenseLayer(network, n_units=1024, act=tf.nn.tanh, name='g/fc')
    # model.add(Dense(128*7*7))
    # model.add(BatchNormalization())
    # model.add(Activation('tanh'))
    network = tl.layers.DenseLayer(network, n_units=128 * 7 * 7, act=tf.identity, name='g/fc2')
    # TODO: We might need to reshape first. https://github.com/zsdonghao/dcgan/blob/master/main.py
    network = tl.layers.BatchNormLayer(network, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='g/bn')
    network.outputs = tf.nn.tanh(network.outputs, name='g/bh_tanh')
    
    # model.add(Reshape((128, 7, 7), input_shape=(128*7*7,)))
    network = tl.layers.ReshapeLayer(network, shape=(-1, 7, 7, 128), name='g/reshape')
    
    # model.add(UpSampling2D(size=(2, 2)))
    # model.add(Convolution2D(64, 5, 5, border_mode='same'))
    # model.add(Activation('tanh'))
    # In TL we just do de-conv2d.
    network = tl.layers.DeConv2dLayer(network, shape=[5, 5, 64, 128], output_shape=[FLAGS.batch_size, 14, 14, 64], strides=[1, 2, 2, 1], act=tf.identity, name='g/deconv1')
    network = tl.layers.BatchNormLayer(network, act=tf.nn.tanh, is_train=is_train, name='g/bn_after_deconv1')
    # model.add(UpSampling2D(size=(2, 2)))
    # model.add(Convolution2D(1, 5, 5, border_mode='same'))
    # model.add(Activation('tanh'))
    # ditto
    network = tl.layers.DeConv2dLayer(network, shape=[5, 5, 1, 64], output_shape=[FLAGS.batch_size, 28, 28, 1], strides=[1, 2, 2, 1], act=tf.identity, name='g/deconv2')
    network = tl.layers.BatchNormLayer(network, act=tf.nn.tanh, is_train=is_train, name='g/bn_after_deconv2')
    return network.outputs

def DiscriminativeNet(X):
  flat = tf.reshape(X, shape=(-1, IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL))
  h1 = Layer('h1').relu_hidden_layer(flat, 1000)
  h2 = Layer('h2').relu_hidden_layer(h1, 400)
  return Layer('Y_pred').linear_layer(h2, 1)

def DiscriminativeConvolutionNet(X, is_train, reuse):
  with tf.variable_scope("Discriminative", reuse=reuse):
    tl.layers.set_name_reuse(reuse)

    network = tl.layers.InputLayer(X, name='d/X')
    # model.add(Convolution2D(64, 5, 5, border_mode='same', input_shape=(1, 28, 28)))
    # model.add(Activation('tanh'))
    network = tl.layers.Conv2d(network, 64, (5, 5), (1, 1), act=tf.nn.tanh, name='d/conv1')
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    network = tl.layers.MaxPool2d(network, (2, 2), padding='SAME', name='d/pool1')
    
    # model.add(Convolution2D(128, 5, 5))
    # model.add(Activation('tanh'))
    network = tl.layers.Conv2d(network, 128, (5, 5), (1, 1), act=tf.nn.tanh, name='d/conv2')
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    network = tl.layers.MaxPool2d(network, (2, 2), padding='SAME', name='d/pool2')
    # model.add(Flatten())
    network = tl.layers.FlattenLayer(network, name='d/flat')
    # model.add(Dense(1024))
    # model.add(Activation('tanh'))
    network = tl.layers.DenseLayer(network, n_units=1024, act=tf.nn.tanh, name='d/fc')
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    network = tl.layers.DenseLayer(network, n_units=1, act=tf.identity, name='d/pred')
    return network.outputs
  

def main():
  mnist_data = load_data()
  def next_image_batch():
    images, _ = mnist_data.train.next_batch(FLAGS.batch_size)
    return images.reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
   
  graph = tf.Graph()
  with graph.as_default():
    with tf.Session() as session:
      # Construct G.
      prior_placeholder = tf.placeholder(tf.float32, shape=[None, PRIOR_DIM], name='prior')
      X_gen_in_G_train = GenerativeConvolutionNet(prior_placeholder, is_train=True, reuse=False)
      X_gen_in_D_train = GenerativeConvolutionNet(prior_placeholder, is_train=False, reuse=True)
      Y_gen_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name='Y_gen')

      # Construct D. With variable sharing.
      X_placeholder = tf.placeholder(tf.float32,
                shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], name='X')
      Y_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

      #Y_gen_pred = DiscriminativeNet(X_gen) # DiscriminativeConvolutionNet(X_gen)
      Y_gen_pred_in_G_train = DiscriminativeConvolutionNet(X_gen_in_G_train, is_train=False, reuse=False)
      Y_gen_pred_in_D_train = DiscriminativeConvolutionNet(X_gen_in_D_train, is_train=True, reuse=True)
      # Y_pred = DiscriminativeNet(X_placeholder) # DiscriminativeConvolutionNet(X_placeholder)
      Y_pred = DiscriminativeConvolutionNet(X_placeholder, is_train=True, reuse=True)

      D_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(Y_gen_pred_in_D_train, Y_gen_placeholder)) + \
               tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(Y_pred, Y_placeholder))
      G_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(Y_gen_pred_in_G_train, Y_gen_placeholder))
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
          X_data = next_image_batch()
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

        if iter_idx % 100 == 0 and iter_idx > 0:
          saver.save(session, os.path.join(FLAGS.save_dir, 'model.ckpt'), global_step=iter_idx)
          # Generate a couple images and save locally.
          prior_sample = sample_from_prior()
          X_gen_values = session.run(X_gen_in_D_train, feed_dict={prior_placeholder: prior_sample})
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
