#coding=utf-8
import tensorflow as tf 
import numpy as np 
#constant(hello world),variable,session两种方式，用梯度下降拟合曲线，构建简易的神经网络层
def constant():
	output = tf.constant('hello, tensorflow!')
	sess = tf.Session();
	print sess.run(output)
	m1 = tf.constant([[1,2]])
	m2 = tf.constant([[2,2],[3,3]])
	m3 = tf.matmul(m1,m2)
	print sess.run(m3)
	sess.close()
#constant();

def variable():
	v1 = tf.Variable(0)
	newV1 = tf.add(v1,1)
	update = tf.assign(v1,newV1)
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		for i in range(0,10):
			#sess.run(newV1)
			sess.run(update)
		print sess.run(v1)

#variable()
x1 = tf.constant([[1,2],[3,3]])
x2 = tf.constant([[3,4],[4,4]])
x3 = tf.reduce_sum(tf.square(x2-x1),0)
with tf.Session() as sess:
	print sess.run(x3)