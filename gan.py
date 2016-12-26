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


def linear_layer(X, hidden_unit_num, prefix):
  W = tf.get_variable(prefix + "_weights", [X.get_shape().as_list()[-1], hidden_unit_num], \
      initializer=tf.random_normal_initializer())
  b = tf.get_variable(prefix + "_bias", [hidden_unit_num], \
      initializer=tf.constant_initializer(0.0))
  return tf.matmul(X, W) + b


def relu_hidden_layer(input_vec, hidden_unit_num, prefix):
  return tf.nn.relu(linear_layer(input_vec, hidden_unit_num, prefix))


def sigmoid_hidden_layer(input_vec, hidden_unit_num, prefix):
  return tf.nn.sigmoid(linear_layer(input_vec, hidden_unit_num, prefix))


PRIOR_DIM  = 100
IMAGE_SIZE = 784
LABEL_NUM = 2

def sample_from_prior():
  return np.random.standard_normal((FLAGS.batch_size, PRIOR_DIM))

def GenerativeNet(prior):
  h1 = relu_hidden_layer(prior, 2000, 'h1')
  h2 = relu_hidden_layer(h1, 1000, 'h2')
  X_gen = sigmoid_hidden_layer(h2, IMAGE_SIZE, 'X_gen')
  return X_gen

def DiscriminativeNet(X):
  h1 = relu_hidden_layer(X, 1000, 'h1')
  h2 = relu_hidden_layer(h1, 400, 'h2')
  return linear_layer(h2, LABEL_NUM, 'Y_pred')

def main():
  mnist_data = load_data()
  images, labels = mnist_data.train.images, mnist_data.train.labels
  label_to_images, label_to_pointer = {}, {}
  for l in range(LABEL_NUM):
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
    batch = imgs[p: new_p, :]
    return batch
   
  # Construct G.
  graph = tf.Graph()
  with graph.as_default():
    with tf.Session() as session:
      prior_placeholder = tf.placeholder(tf.float32, shape=[None, PRIOR_DIM], name='prior')
      with tf.variable_scope('Generative'):
        X_gen = GenerativeNet(prior_placeholder)
        Y_gen_placeholder = tf.placeholder(tf.int32, shape=[None, LABEL_NUM], name='Y_gen')

      # Construct D. With variable sharing.
      with tf.variable_scope('Discriminative'):
        X_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE], name='X')
        Y_placeholder = tf.placeholder(tf.int32, shape=[None, LABEL_NUM], name='Y')
        with tf.variable_scope('DiscriminativeShared') as share_scope:
          Y_gen_pred = DiscriminativeNet(X_gen)
          share_scope.reuse_variables()
          Y_pred = DiscriminativeNet(X_placeholder)

      D_loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(Y_gen_pred, Y_gen_placeholder)) + \
               tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(Y_pred, Y_placeholder))
      G_loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(Y_gen_pred, Y_gen_placeholder))
      print(graph.get_all_collection_keys())
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
            Y_placeholder: np.asarray([[0, 1] for _ in range(FLAGS.batch_size)]),  # 1 means True
            prior_placeholder: prior_sample,
            Y_gen_placeholder: np.asarray([[1, 0] for _ in range(FLAGS.batch_size)]),
          })

        prior_sample = sample_from_prior()
        session.run(G_train, feed_dict={
          prior_placeholder: prior_sample,
          Y_gen_placeholder: np.asarray([[0, 1] for _ in range(FLAGS.batch_size)]),
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
      default=64,
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
