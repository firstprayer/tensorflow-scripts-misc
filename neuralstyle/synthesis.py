from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import time
import os
from scipy import misc

FLAGS = None

def load_inception():
  with tf.gfile.FastGFile(FLAGS.model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  t_input = tf.placeholder(np.float32, name='input', shape=[None, 224, 224, 3]) # define the input tensor
  imagenet_mean = 117.0
  t_preprocessed = t_input - imagenet_mean
  for node in graph_def.node:
    if node.op != 'Const':
      print(node.name, node.op) 
  # print(graph_def)
  print(tf.import_graph_def(graph_def, {'input':t_preprocessed}))
  return t_preprocessed


def load_vgg16():
  with open(FLAGS.model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    s = f.read()
    graph_def.ParseFromString(s)

  t_input = tf.placeholder(np.float32, name='input', shape=[None, 224, 224, 3]) # define the input tensor
  for node in graph_def.node:
    if node.op != 'Const':
      print(node.name, node.op) 
  tf.import_graph_def(graph_def, {'input': t_input})
  return t_input
  


def T(graph, layer):
  '''Helper for getting layer output tensor'''
  return graph.get_tensor_by_name("import/%s:0" % layer)


def main():
  graph = tf.Graph()
  with graph.as_default():
    session = tf.InteractiveSession(graph=graph)
    # Load pretrained model
    X = load_inception()

    # print(graph.get_all_collection_keys())
    print(T(graph, 'conv2d0_w'))
    # print(T(graph, 'mixed4d_3x3_bottleneck_pre_relu'))

    content_image = misc.imread(FLAGS.content_image)
    content_image = misc.imresize(content_image, size=(224, 224, 3))
    content_image = np.expand_dims(content_image, 0)

    style_image = misc.imread(FLAGS.style_image)
    style_image = misc.imresize(style_image, size=(224, 224, 3))
    style_image = np.expand_dims(style_image, 0)
    
    content_layers = [
      'conv2d0',
      'conv2d1',
      'conv2d2',
      'conv2d0_pre_relu/conv',
      'conv2d1_pre_relu/conv',
      'conv2d2_pre_relu/conv',
      # 'mixed4d_3x3_bottleneck_pre_relu/conv',
      # 'mixed3a_5x5',
      # 'mixed3b_5x5',
      # 'mixed4a_5x5',
      # 'mixed4b_5x5',
      # 'mixed5a_5x5',
    ]
    for layer_name in content_layers:
      t_layer = T(graph, layer_name)
      content_activation = session.run(t_layer, feed_dict={X: content_image})
      t_content_loss = tf.reduce_mean(tf.square(t_layer - content_activation))

      t_layer_shape = t_layer.get_shape().as_list()
      feature_map_size = t_layer_shape[1] * t_layer_shape[2]
      t_layer_vectorized = tf.reshape(t_layer,
              shape=[feature_map_size, t_layer_shape[-1]])
      t_gram_mat = tf.matmul(t_layer_vectorized, t_layer_vectorized, transpose_a=True)
      style_activation = session.run(t_gram_mat, feed_dict={X: style_image})
      t_style_loss = tf.reduce_mean(tf.square(t_gram_mat - style_activation)) \
              / (4 * feature_map_size * feature_map_size)

      t_total_loss = FLAGS.alpha * t_content_loss + (1 - FLAGS.alpha) * t_style_loss

      img_syn = np.random.uniform(size=(1, 224, 224, 3)) + 100.0
      # opt = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.step)
      # t_grads_and_vals = opt.compute_gradients(t_loss, [X])
      t_grad = tf.gradients(t_total_loss, X)[0]

      def check_point(num_iter, img):
        misc.imsave(FLAGS.output_dir + '/%s_syn_%d.jpg' % (layer_name.replace('/', '-'), num_iter), img)

      for k in range(FLAGS.iter_num + 1):
        # grads_and_vals = session.run(t_grads_and_vals, feed_dict={X: img_syn})
        # print(grads_and_vals)
        grad, loss = session.run([t_grad, t_total_loss], feed_dict={X: img_syn})
        grad /= grad.std() + 1e-8
        img_syn -= grad * FLAGS.step
        if k % 100 == 0 and k:
          print(k, loss)
          check_point(k, img_syn.squeeze())


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_path',
      type=str,
      default='tensorflow_inception_graph.pb',
      help='Path to pretrained model',
  )
  parser.add_argument(
      '--content_image',
      type=str,
      default='content.jpg',
      help='Path for content image',
  )
  parser.add_argument(
      '--style_image',
      type=str,
      default='style.jpg',
      help='Path for style image',
  )
  parser.add_argument(
      '--step',
      type=float,
      default=1.0,
      help='Learning rate',
  ) 
  parser.add_argument(
      '--alpha',
      type=float,
      default=0.9,
      help='Weight on the content loss',
  ) 
  parser.add_argument(
      '--iter_num',
      type=int,
      default=20,
      help='Learning iter',
  ) 
  parser.add_argument(
      '--output_dir',
      type=str,
      default='syn_images',
      help='Output directory for images',
  ) 
  FLAGS, _ = parser.parse_known_args()
  main()
