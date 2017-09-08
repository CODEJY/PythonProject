#coding=utf-8
import input_data
import tensorflow as tf
import numpy as np

def varWeight(shape):
	init = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(init)

def varBiases(shape):
	init = tf.constant(0.1,shape=shape)
	return tf.Variable(init)
#卷积层，提取特征
def conv2d(x,w):
	return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
#池化层，降低维度，减少参数，提高效率
def maxPool_2x2(conv_x):
	return tf.nn.max_pool(conv_x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#添加卷积层+激活层
def add_conv(in_layer,wShape,bShape):
	weights = varWeight(wShape)
	biases = varBiases(bShape)
	conv1 = tf.nn.relu(conv2d(in_layer,weights)+biases)
	return conv1

#添加第一个全连接层
def add_fc1(in_layer,wShape,bShape):
	pool2_reshape = tf.reshape(in_layer,[-1,7*7*64])
	weights = varWeight(wShape)
	biases = varBiases(bShape)
	fc_layer = tf.nn.relu(tf.matmul(pool2_reshape,weights)+biases)
	fc_layer_drop = tf.nn.dropout(fc_layer,drop)
	return fc_layer_drop

def add_output(in_layer,wShape,bShape):
	weights = varWeight(wShape)
	biases = varBiases(bShape)
	prediction = tf.nn.softmax(tf.matmul(in_layer,weights)+biases)
	return prediction
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
learning_rate = 1e-4
x_in = tf.placeholder(tf.float32,[None,784])
y_in = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(x_in, [-1, 28, 28, 1])
#drop = tf.placeholder(tf.float32)
drop = tf.Variable(tf.constant(0.8))
#第一层卷积、激活、池化层
conv1 = add_conv(x_image,[5,5,1,32],[32])
pool1 = maxPool_2x2(conv1)
#第二层卷积、激活、池化层
conv2 = add_conv(pool1,[5,5,32,64],[64])
pool2 = maxPool_2x2(conv2)
#全连接层

fc_layer_drop = add_fc1(pool2,[7*7*64,1024],[1024])
#预测的结果
y_pre = add_output(fc_layer_drop,[1024,10],[10])
#交叉熵
cross_entropy = -tf.reduce_sum(y_in*tf.log(y_pre),1)
#训练
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
#准确率
correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y_in,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(1000):
		batch_xs,batch_ys = mnist.train.next_batch(200)
		sess.run(train_step,feed_dict={x_in: batch_xs, y_in: batch_ys})
		if i % 100 == 0:
			print '精确度：%f' % (accuracy.eval(feed_dict={x_in: mnist.test.images, y_in: mnist.test.labels}))
	print '精确度：%f' % (accuracy.eval(feed_dict={x_in: mnist.test.images, y_in: mnist.test.labels}))