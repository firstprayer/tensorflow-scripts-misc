from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import time
import os
import scipy
from scipy import misc
# from vgg import vgg19

FLAGS = None
IMAGENET_MEAN = 117.0
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


def load_vgg19():
  model = vgg19.Vgg19(FLAGS.model_path)

  t_input = tf.get_variable(
          name='input', \
          shape=[1, FLAGS.img_height, FLAGS.img_width, 3], \
          initializer=tf.random_normal_initializer())
  model.build(t_input / 255.0)
  return t_input


def load_vgg19_v2():
  from neural_style import vgg

  # model = vgg19.Vgg19(FLAGS.model_path)

  t_input = tf.get_variable(
          name='input', \
          shape=[1, FLAGS.img_height, FLAGS.img_width, 3], \
          initializer=tf.random_normal_initializer())

  red, green, blue = tf.split(3, 3, t_input)
  print(red.get_shape().as_list(), green.get_shape().as_list(), blue.get_shape().as_list())
  normalized_input = tf.concat(3, [red - 123.68, green - 116.779, blue - 103.939])
  vgg.net('imagenet-vgg-verydeep-19.mat', normalized_input)
  return t_input


def print_graph_node_names(graph):
  '''For printing out graph structure'''
  for node in graph.as_graph_def().node:
    # if node.op == 'Const':
    print(node.name, node.op) 


def T(graph, layer):
  '''Helper for getting layer output tensor'''
  return graph.get_tensor_by_name("%s:0" % layer)


def imread(path):
  img = scipy.misc.imread(path).astype(np.float)
  img = misc.imresize(img, size=(FLAGS.img_height, FLAGS.img_width, 3)).astype(np.float)
  if len(img.shape) == 2:
    img = np.dstack((img,img,img))
  img = np.expand_dims(img, 0)
  return img


def get_activation():
  with tf.Graph().as_default() as graph:
    session = tf.InteractiveSession(graph=graph)
    X = load_vgg19_v2()

    style_image = imread(FLAGS.style_image)
    
    style_layers = [
      # VGG19 layers
      'Relu',
      'Relu_2',
      'Relu_4',
      'Relu_8',
      'Relu_12',
    ]
    layer_style_loss_list = []
    session.run(tf.global_variables_initializer())
    session.run(X.assign(style_image)) 
    for layer_name in style_layers:
      t_layer = T(graph, layer_name)
      _, height, width, channel = t_layer.get_shape().as_list()
      t_layer_vectorized = tf.reshape(t_layer,
              shape=[-1, channel])
      size = height * width * channel
      t_gram_mat = tf.matmul(t_layer_vectorized, t_layer_vectorized, transpose_a=True) / size
      style_activation = session.run(t_gram_mat)


def main():
  graph = tf.Graph()
  with graph.as_default():
    session = tf.InteractiveSession(graph=graph)
    # Load pretrained model
    X = load_vgg19_v2()
    # print_graph_node_names(graph)

    content_image = imread(FLAGS.content_image)
    style_image = imread(FLAGS.style_image)
    
    style_layers = [
      # VGG19 layers
      'Relu',
      'Relu_2',
      'Relu_4',
      'Relu_8',
      'Relu_12',
      # 'BiasAdd', # 'conv1_1/Relu',
      # 'BiasAdd_2', # 'conv2_1/Relu',
      # 'BiasAdd_4', # 'conv3_1/Relu',
      # 'BiasAdd_8', # 'conv4_1/Relu',
      # 'BiasAdd_12', # 'conv5_1/Relu',
    ]
    layer_style_loss_list = []
    session.run(tf.global_variables_initializer())
    session.run(X.assign(style_image)) 
    for layer_name in style_layers:
      t_layer = T(graph, layer_name)
      _, height, width, channel = t_layer.get_shape().as_list()
      t_layer_vectorized = tf.reshape(t_layer,
              shape=[-1, channel])
      size = height * width * channel
      t_gram_mat = tf.matmul(
        tf.transpose(t_layer_vectorized),
        t_layer_vectorized,
      ) / size
      style_activation = t_gram_mat.eval()
      # print(layer_name, style_activation.flatten()[: 20])
      t_style_loss = 2 * tf.nn.l2_loss(t_gram_mat - style_activation) / style_activation.size
      layer_style_loss_list.append(t_style_loss)
     
    content_layers = [
      # VGG19
      # 'MaxPool',
      # 'pool1',
      'Relu_9',
    ]
    for layer_name in content_layers:
      print('=================Running layer', layer_name, 'as content layer')
      t_layer = T(graph, layer_name)
      session.run(X.assign(content_image))
      content_activation = session.run(t_layer)
      t_content_loss = 2 * tf.nn.l2_loss(t_layer - content_activation) / content_activation.size

      for i in [4]: #reversed(range(5)): # reversed([1, 4, 7, 10, 13]):
        print('Running', style_layers[: i + 1], 'as style layers')

        t_total_loss = FLAGS.alpha * t_content_loss
        for t_style_loss in layer_style_loss_list[: i + 1]:
            t_total_loss += (1 - FLAGS.alpha) * t_style_loss
        grad_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) \
                          .minimize(t_total_loss, var_list=[X])
        # Initialize variables needed by Adam
        session.run(tf.global_variables_initializer())

        def get_file_save_path(iter_num):
          return FLAGS.output_dir + '/%s_syn_%d(%d).jpg' \
                  % (layer_name.replace('/', '-'), iter_num, i)
        
        def get_model_save_path(iter_num=None):
          path = FLAGS.save_dir + '/model-%s_syn(%d).cpkt' \
                  % (layer_name.replace('/', '-'), i)
          if iter_num is not None:
            path += '-%d' % iter_num
          return path

        saver = tf.train.Saver()
        print('initial path: ', FLAGS.initial)
        if FLAGS.resume_iter > 0:
          saver.restore(
            session,
            get_model_save_path(FLAGS.resume_iter))
        else:
          if FLAGS.initial:
            initial = imread(FLAGS.initial)
          else:
            initial = np.random.normal(size=(1, FLAGS.img_height, FLAGS.img_width, 3), \
                  scale=np.std(content_image) * 0.1) * 0.256
          # initial = np.random.uniform(size=(1, FLAGS.img_height, FLAGS.img_width, 3))
          session.run(X.assign(initial))
          print(X.eval().flatten()[: 20])

        print('Loss before starting: ', t_total_loss.eval())
          
        def check_point(num_iter):
          img = session.run(X).squeeze()
          print(img.flatten()[: 20])
          misc.imsave(get_file_save_path(num_iter), img)
          saver.save(session, get_model_save_path(), global_step=num_iter)

        start_iter = 0
        if FLAGS.resume_iter > 0:
          start_iter = FLAGS.resume_iter + 1
        for k in range(start_iter, FLAGS.iter_num + 1):
          _, loss, content_loss = session.run([grad_op, t_total_loss, t_content_loss]) 
          # if k % 10 == 0 and k:
          print(k, loss, content_loss * FLAGS.alpha, content_loss)

          if k % 10 == 0 and k:
            check_point(k)


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
      '--initial',
      type=str,
      default='',
      help='Initiation for optimization',
  )
  parser.add_argument(
      '--learning_rate',
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
      '--img_width',
      type=int,
      default=224,
      help='Image width',
  ) 
  parser.add_argument(
      '--img_height',
      type=int,
      default=224,
      help='Image height',
  ) 
  parser.add_argument(
      '--resume_iter',
      type=int,
      default=0,
      help='If positive, resume from previous training by loading previous result',
  ) 
  parser.add_argument(
      '--output_dir',
      type=str,
      default='syn_images',
      help='Output directory for images',
  )
  parser.add_argument(
      '--save_dir',
      type=str,
      default='models',
      help='Output directory for models',
  )
  FLAGS, _ = parser.parse_known_args()
  main()
  # get_activation()
