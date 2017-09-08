#coding=utf-8
import input_data
import tensorflow as tf
import numpy as np

def varWeight(shape,name=None):
	init = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(init,name=name)

def varBiases(shape,name=None):
	init = tf.constant(0.1,shape=shape)
	return tf.Variable(init,name=name)
#卷积层，提取特征
def conv2d(x,w,name=None):
	return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME',name=name)
#池化层，降低维度，减少参数，提高效率
def maxPool_2x2(conv_x,name=None):
	return tf.nn.max_pool(conv_x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)

def def_var(rate=None,optimizer_func=None,drop_=1.0):
	global learning_rate
	global optimizer
	global drop
	global cross_entropy
	global train_step
	global fc1_layer
	global fc1_layer_drop
	d = tf.constant(drop_)
	drop = tf.Variable(d)
	fc1_layer_drop = tf.nn.dropout(fc1_layer,drop)
	learning_rate = rate
	optimizer = optimizer_func
	train_step = optimizer(learning_rate).minimize(cross_entropy)


def make_hparam_string(rate=None,func=None,drop_=1.0):
	if func == tf.train.GradientDescentOptimizer:
		return '%s,GradientDescent,drop=%f'%(rate,drop_)
	else:
		return '%s,Adam,drop=%f'%(rate,drop_)

#准备数据
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
drop = tf.Variable(tf.constant(1.0),tf.float32)
learning_rate = 1E-2
optimizer = tf.train.GradientDescentOptimizer
with tf.name_scope('Inputs'):
	x_in = tf.placeholder(tf.float32,[None,784],name='x_in')
	y_in = tf.placeholder(tf.float32,[None,10],name='y_in')
	x_image = tf.reshape(x_in, [-1, 28, 28, 1],name='x_image')
	#drop = tf.placeholder(tf.float32,name='drop')
#第一层卷积、激活、池化层
with tf.name_scope('Conv1'):
	w_conv1 = varWeight([5,5,1,32],name='w_conv1')
	b_conv1 = varBiases([1,32],name='b_conv1')
	conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1,name='conv1')
with tf.name_scope('Pool1'):
	pool1 = maxPool_2x2(conv1,name='MaxPool_2x2')

#第二层卷积、激活、池化层
with tf.name_scope('Conv2'):
	w_conv2 = varWeight([5,5,32,64],name='w_conv2')
	b_conv2 = varBiases([1,64],name='b_conv2')
	conv2 = tf.nn.relu(conv2d(pool1,w_conv2)+b_conv2,name='conv2')
with tf.name_scope('Pool2'):
	pool2 = maxPool_2x2(conv2)

#全连接层
with tf.name_scope("Fully_Connected_Layer"):
	pool2_reshape = tf.reshape(pool2,[-1,7*7*64],name='reshape')
	w_fc1 = varWeight([7*7*64,1024],name='w_fc1')
	b_fc1 = varBiases([1,1024],name='b_fc1')
	fc1_layer = tf.nn.relu(tf.matmul(pool2_reshape,w_fc1)+b_fc1)
	fc1_layer_drop = tf.nn.dropout(fc1_layer,drop,name='fc1_layer_drop')

#预测的结果
with tf.name_scope('Output'):
	w_output = varWeight([1024,10],name='w_output')
	b_output = varBiases([1,10],name='b_output')
	y_pre = tf.nn.softmax(tf.matmul(fc1_layer_drop,w_output)+b_output,name='y_pre')
with tf.name_scope('Training'):
#交叉熵
	cross_entropy = -tf.reduce_sum(y_in*tf.log(y_pre),1)
#训练
	train_step = optimizer(learning_rate).minimize(cross_entropy)
#准确率
with tf.name_scope('Accuracy'):
	correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y_in,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy',accuracy)

with tf.Session() as sess:
	for drop_ in [0.7,0.5,0.4]:
		for rate in [1E-5,1E-4,1E-3]:
			for func in [tf.train.GradientDescentOptimizer,tf.train.AdamOptimizer]:	
				def_var(rate,func,drop_)
				hparam_str = make_hparam_string(learning_rate,optimizer,drop_)
				print hparam_str
				sess.run(tf.global_variables_initializer())
				writter = tf.summary.FileWriter('Compare_cnnMNIST/'+hparam_str,sess.graph)
				merge_all = tf.summary.merge_all()
				for i in range(3000):
					batch_xs,batch_ys = mnist.train.next_batch(100)
					sess.run(train_step,feed_dict={x_in: batch_xs, y_in: batch_ys})
					if i % 100 == 0:
						rs = sess.run(merge_all,feed_dict={x_in:batch_xs,y_in:batch_ys})
						writter.add_summary(rs,i)
						#print '精确度：%f' % (accuracy.eval(feed_dict={x_in: mnist.test.images, y_in: mnist.test.labels}))
				print '精确度：%f' % (accuracy.eval(feed_dict={x_in: mnist.test.images, y_in: mnist.test.labels}))