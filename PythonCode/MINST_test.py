#coding=utf-8
import input_data
import tensorflow as tf
import numpy as np
mnist = input_data.read_data_sets("MINST_data/",one_hot=True)
learning_rate = 0.01
with tf.name_scope('Inputs'):
	x_in = tf.placeholder(tf.float32,[None,784],name='x_in')
	y_in = tf.placeholder(tf.float32,[None,10],name='y_in')
with tf.name_scope('input_reshape'):  
    image_shaped_input = tf.reshape(x_in, [-1, 28, 28, 1])  
    tf.summary.image('input_image',image_shaped_input, 10)
def def_var(rate=None,init_func = None):
	global learning_rate 
	global weights
	global biases
	weights = tf.Variable(init_func([784,10]),name='weights')
	biases = tf.Variable(init_func([1,10]),name='biases')
	learning_rate = rate
with tf.name_scope('Outputs'):
	with tf.name_scope('Weights'):
		weights = tf.Variable(tf.zeros([784,10]),name='weights')
		tf.summary.histogram('weights',weights)
	with tf.name_scope('Biases'):
		biases = tf.Variable(tf.zeros([1,10]),name='biases')
		tf.summary.histogram('biases',biases)
	y_pre = tf.nn.softmax(tf.matmul(x_in,weights)+biases)
with tf.name_scope('Cross_entropy'):
	cross_entropy = -tf.reduce_sum(y_in*tf.log(y_pre))
	tf.summary.scalar('cross_entropy',cross_entropy)
with tf.name_scope('Train'):
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
with tf.name_scope('Accuracy'):
	correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y_in,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy',accuracy)
def make_hparam_string(rate=None,func=None):
	if func == tf.zeros:
		return '%s,tf.zeros'%(rate)
	else:
		return '%s,tf.random_normal'%(rate)
with tf.Session() as sess:
	for rate in [1E-2,1E-3]:
		for func in [tf.zeros,tf.random_normal]:
			hparam_str = make_hparam_string(rate,func)
			def_var(rate,func)
			writter = tf.summary.FileWriter('mnist/'+hparam_str,sess.graph)
			merge_all = tf.summary.merge_all()
			sess.run(tf.global_variables_initializer())
			for i in range(1001):
				'''
				if i == 0:
					print sess.run(weights)
					print sess.run(biases)
					print learning_rate
				'''
				batch_xs,batch_ys = mnist.train.next_batch(50)
				if i % 50 == 0:
					rs = sess.run(merge_all,feed_dict={x_in:batch_xs,y_in:batch_ys})
					writter.add_summary(rs,i)
				sess.run(train_step,feed_dict={x_in: batch_xs, y_in: batch_ys})
			print '%s  精确度：%f' % (hparam_str,accuracy.eval(feed_dict={x_in: mnist.test.images, y_in: mnist.test.labels}))