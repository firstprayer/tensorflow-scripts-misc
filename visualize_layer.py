from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import time

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

MODEL_PATH = './model.ckpt-14999'

session = tf.Session()
saver = tf.train.import_meta_graph('./model.ckpt-14999.meta')
saver.restore(session, MODEL_PATH)

print(session.graph.get_all_collection_keys())
