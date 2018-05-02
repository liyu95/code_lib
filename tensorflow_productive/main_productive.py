#!/usr/bin/env python
import time
import numpy as np
import tflearn
from sklearn.cross_validation import train_test_split
import random
import cPickle
import copy
import math
import sys
import tensorflow as tf
import os
from evaluate_model import *
import pdb
from level_1_model_graph import model_graph
from utils import *
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

start=time.time()

#Global variable, MAX_LENGTH is the maximum length of all sequence.
DROPOUT=True
MAX_LENGTH=1576
TYPE_OF_AA=23
DOMAIN=16306
LOAD=False
train_ratio=0.9
level=0
output_step=100
batch_size = 20
train_steps = 20000



config = tf.ConfigProto()
config.log_device_placement=False
config.allow_soft_placement=True
config.gpu_options.allow_growth=True


class ARG_prediction(object):
	"""load the graph and make prediction"""
	def __init__(self, loc):
		super(ARG_prediction, self).__init__()
		
		self.graph = tf.Graph()
		self.sess = tf.Session(graph=self.graph, config=config)
		with self.graph.as_default():
			saver = tf.train.import_meta_graph(loc + '.meta',
			                                   clear_devices=True)
			saver.restore(self.sess, loc)
			init = tf.global_variables_initializer()
			self.sess.run(init)
			saver.restore(self.sess, loc)

			self.predicted_label = self.graph.get_operation_by_name(
				'accuracy/correct_prediction/ArgMax_1').outputs[0]
			self.pssm=self.graph.get_operation_by_name(
				'placeholder/Placeholder').outputs[0]
			self.encoding=self.graph.get_operation_by_name(
				'placeholder/Placeholder_1').outputs[0]
			self.y_=self.graph.get_operation_by_name(
				'placeholder/Placeholder_2').outputs[0]
			self.domain=self.graph.get_operation_by_name(
				'placeholder/Placeholder_3').outputs[0]
			self.keep_prob=self.graph.get_operation_by_name(
				'placeholder/Placeholder_4').outputs[0]

	def whole_set_check(self, data):
		test_pssm = data[0]
		test_encoding = data[1]
		test_funcd = data[2]
		z = np.zeros([batch_size, y_.shape.as_list()[-1]])
		predict_test_label=[]

		number_of_full_batch=int(math.floor(len(test_pssm)/batch_size))
		for i in range(number_of_full_batch):
			predicted_label_out = self.sess.run(self.predicted_label,
			    feed_dict={self.pssm: test_pssm[i*batch_size:(i+1)*batch_size], 
			    self.encoding: test_encoding[i*batch_size:(i+1)*batch_size], 
			    self.domain: test_funcd[i*batch_size:(i+1)*batch_size], 
			    self.y_: z, 
			    self.keep_prob: 1.0})
			predict_test_label+=list(predicted_label_out)
			if i%100==0:
				print("working {}".format(i))

		predicted_label_out = self.sess.run(self.predicted_label,
		    feed_dict={self.pssm: test_pssm[number_of_full_batch*batch_size:], 
		    self.encoding: test_encoding[number_of_full_batch*batch_size:], 
		    self.domain: test_funcd[number_of_full_batch*batch_size:], 
		    self.y_: z, 
		    self.keep_prob: 1.0})

		predict_test_label+=list(predicted_label_out)
		return predict_test_label


if __name__ == '__main__':
	#load all data
	data_all = load_level_0_data()
	feed_data = data_all[7:]
	loc='../model/model_level_0.ckpt'
	model_0 = ARG_prediction(loc)
	level_0_prediction = model_0.whole_set_check(feed_data)