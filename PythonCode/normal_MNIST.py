#coding=utf-8
import input_data
import tensorflow as tf
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
learning_rate
init_func = tf.zeros
with tf.name_scope('Inputs'):
	x_in = tf.placeholder(tf.float32,[None,784],name='x_in')
	y_in = tf.placeholder(tf.float32,[None,10],name='y_in')
with tf.name_scope('input_reshape'):  
    image_shaped_input = tf.reshape(x_in, [-1, 28, 28, 1])  
    tf.summary.image('input_image',image_shaped_input, 10)
def def_var(rate=None,func = None):
	global learning_rate 
	global init_func
	global cross_entropy
	global train_step
	init_func = func
	learning_rate = rate
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
with tf.name_scope('Outputs'):
	w_output = tf.Variable(init_func([784,10]),name='w_output')
	tf.summary.histogram('w_output',w_output)
	b_output = tf.Variable(init_func([1,10]),name='b_output')
	tf.summary.histogram('b_output',b_output)
	y_pre = tf.nn.softmax(tf.matmul(x_in,w_output)+b_output)
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
			def_var(rate,func)
			hparam_str = make_hparam_string(learning_rate,init_func)
			writter = tf.summary.FileWriter('normal_MNIST/'+hparam_str,sess.graph)
			merge_all = tf.summary.merge_all()
			sess.run(tf.global_variables_initializer())
			for i in range(1001):
				'''
				if i == 1:
					print sess.run(w_output)
					print sess.run(b_output)
					print learning_rate
				'''
				batch_xs,batch_ys = mnist.train.next_batch(50)
				if i % 50 == 0:
					rs = sess.run(merge_all,feed_dict={x_in:batch_xs,y_in:batch_ys})
					writter.add_summary(rs,i)
				sess.run(train_step,feed_dict={x_in: batch_xs, y_in: batch_ys})
			print '%s  精确度：%f' % (hparam_str,accuracy.eval(feed_dict={x_in: mnist.test.images, y_in: mnist.test.labels}))