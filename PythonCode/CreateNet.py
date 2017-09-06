#coding=utf-8
import tensorflow as tf 
import numpy as np 
#创建层
def add_layer(inputs,in_size,out_size,layer_name='Layer_Default',activation_function=None):
	#layer_name = 'layer%s' % n_layer
	with tf.name_scope(layer_name):
		with tf.name_scope('Weights'):
			weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
			#tf.summary.scalar('weights',weights)
			tf.summary.histogram('weights',weights)
		with tf.name_scope('Biases'):
			biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='b')
			#tf.summary.scalar('biases',biases)
			tf.summary.histogram('biases',biases)
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.matmul(inputs,weights) + biases

		if activation_function == None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)
		tf.summary.histogram('/outputs',outputs)
	return outputs
#输入数据
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
#添加噪点，是数据更加真实，正态分布随机取值
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
#标准输出数据
y_data = np.square(x_data) - 0.5 + noise
#利用占位符定义输出与输出，等到需要用到时再feed
with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32, [None, 1],name='x_in')
	ys = tf.placeholder(tf.float32, [None, 1],name='y_in')

hidden_layer = add_layer(xs,1,10,'hidden_layer1',tf.nn.relu)
hidden_layer2 = add_layer(hidden_layer,10,10,'hidden_layer2',tf.nn.relu)
output_layer = add_layer(hidden_layer2,10,1,'output_layer',tf.nn.relu)
#reduction_indices=[1]对行求和，0对列求和，返回tuple数组
with tf.name_scope('Loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - output_layer),reduction_indices=[1]))
	tf.summary.scalar('loss',loss)
#训练
with tf.name_scope('train'):	
	train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
with tf.Session() as sess:
	merged = tf.summary.merge_all()
	writter = tf.summary.FileWriter("logs/",sess.graph)
	sess.run(tf.global_variables_initializer())
	for i in range(1000):
		sess.run(train,feed_dict={xs:x_data,ys:y_data})
		if i % 50 == 0:
			print sess.run(loss, feed_dict={xs: x_data, ys: y_data})
			rs = sess.run(merged,feed_dict={xs: x_data, ys: y_data})
			writter.add_summary(rs,i)

##测试如何用scalar显示weights和biases的趋势