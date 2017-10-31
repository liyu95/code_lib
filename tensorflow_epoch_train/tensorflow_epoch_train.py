#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from math import sqrt
import numpy as np
from numpy import zeros
import sys
import re
import math
from tf_model_component import *
from batch_object import batch_object
from sklearn.metrics import accuracy_score

weight_decay = 0
learning_rate = 1e-4
output_step = 10
input_shape = [None, 10]
n_class = 2

# define the graph
# store the weight in the dictionary and the out in the dictionary 
# as well
def model_graph(x):
	weight_dict = dict()
	bias_dict = dict()

	out_dict = dict()
	out_dict[-1] = x

	return out_dict

# define the placeholder, cost function, optimizer and do a random initialization of the graph
def start_sess():

	input_x = tf.placeholder(tf.float32,shape=input_shape)
	y_ = tf.placeholder(tf.float32, [None, n_class])

	out_dict = model_graph(input_x)
	y = out_dict['out']
	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		logits=out_dict['out_logit'], labels=y_))
	weight_collection=[v for v in tf.get_collection(
		tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]
	l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in weight_collection])
	cost = tf.add(cost, weight_decay*l2_loss)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Evaluate model
	predicted_label = tf.argmax(y, 1)
	correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Start the session
	sess = tf.InteractiveSession(config=tf.ConfigProto(
		log_device_placement=False,allow_soft_placement=True))

	# Initializing the variables
	sess.run(tf.global_variables_initializer())
	return sess

def generate_batch(feature, label, batch_size):
    batch_index = randint(0,len(feature),batch_size)
    feature_batch = batch_process(feature, batch_index)
    label_batch = batch_process(label, batch_index)
    return (feature_batch, label_batch)


def set_check(sess, test_feature, test_label, batch_size):
    predict_test_label=[]
    prob_test = []
    number_of_full_batch=int(math.floor(len(test_feature)/batch_size))

    for i in range(number_of_full_batch):
        prob_out, predicted_label_out = sess.run([y, predicted_label],
            feed_dict={x: test_feature[i*batch_size:(i+1)*batch_size],
            y_: test_label[i*batch_size:(i+1)*batch_size]})
        prob_test += list(prob_out)
        predict_test_label+=list(predicted_label_out)
    
    prob_out, predicted_label_out = sess.run([y, predicted_label],
        feed_dict={x: test_feature[number_of_full_batch*batch_size:], 
        y_: test_label[number_of_full_batch*batch_size:]})
    

    prob_test += list(prob_out)
    predict_test_label+=list(predicted_label_out)

    true_label = np.argmax(test_label, axis=1)
    acc_score=accuracy_score(true_label,np.array(predict_test_label))
    print('The accuracy for the whole set is {}'.format(acc_score))

    return (prob_test, predict_test_label)

# train the model on the training data and get the prediction prob
# of the test dataset
def train_and_test(x_train, y_train, x_test, y_test):
	sess = start_sess()

	# define the training and test part
	batch_size = p['batch_size']
	nb_epoch=p['nb_epoch']

	for epoch in range(nb_epoch):
		x_train_obj = batch_object(x_train, batch_size)
		y_train_obj = batch_object(y_train, batch_size)
		for step in range(int(len(x_train)/batch_size)+1):
			x_train_batch = x_train_obj.next_batch()
			y_train_batch = y_train_obj.next_batch()
			sess.run(optimizer, feed_dict={
				input_x: x_train_batch,
				y_: y_train_batch
				})
			if step%output_step ==0:
				loss, acc = sess.run([cost, accuracy], feed_dict = {
					input_x: x_train_batch,
					y_: x_train_batch
				})
				print('Train step %d'%i)
				print('Train loss: %f, train acc: %f'%(loss, acc))
				# test_batch = generate_batch(x_test, y_test, batch_size)
				# loss, acc = sess.run([cost, accuracy], feed_dict ={
				# 	input_x: test_batch[0],
				# 	y_: test_batch[1]
				# 	})
				# print('Test loss: %f, test acc: %f'%(loss, acc))
		print('Result after training epoch {}'.format(epoch))
		set_check(sess, x_train, y_train, batch_size)
		# set_check(sess, x_test, y_test, batch_size)

	prob, test_label_pred = set_check(sess, x_test, y_test, batch_size)
	# define the model storage part
	saver = tf.train.Saver()
	saver.save(sess, './'+p['model_output'])

	return prob

def load_and_test(x_test, y_test):
	sess = start_sess()
	saver = tf.train.Saver()
	saver.restore(sess, './'+p['model_file'])
	batch_size = p['batch_size']
	prob, test_label_pred = set_check(sess, x_test, y_test, batch_size)
	return prob


# If the model exist go to load and test
# If the model does not exist, go to the train and test
if (p['read_model'] == 1):
	predict = train_and_test(X_train, y_train, X_test, y_test)
else:
	predict = load_and_test(X_test, y_test)
