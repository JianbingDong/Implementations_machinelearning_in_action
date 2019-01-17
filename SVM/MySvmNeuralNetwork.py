"""
此函数使用神经网络的形式来对支持向量机进行求解,
利用梯度下降来优化合页损失函数
"""

import tensorflow as tf
import numpy as np
import pickle


def loadDataset():
	"""
	此函数用于创建数据集，仅有三个样本，与《统计学习方法》例7.2的数据一致
	"""
	dataSet = [[3, 3],
			   [4, 3],
			   [1, 1]]
	labelSet = [1, 1, -1]

	return dataSet, labelSet


def train_weights(dataset, labelset):
	"""
	此函数用于训练网络中的参数
	#arguments:
		dataset, labelset: loadDataset函数的返回值
	"""
	m = len(dataset) #样本个数
	n = len(dataset[0]) #特征个数
	data = tf.convert_to_tensor(dataset, dtype=tf.float32) #(3, 2)
	labels = tf.convert_to_tensor(labelset, dtype=tf.float32) #(3,)
	labels = tf.reshape(labels, (m, 1)) #(3, 1)

	#前向传播过程
	weights = tf.Variable(tf.truncated_normal_initializer()(shape=(n, 1)), trainable=True) #(2, 1)
	bias = tf.Variable(tf.constant_initializer(0.1)(shape=(1,)), trainable=True) #(1, )

	logits = tf.matmul(data, weights) + bias #(3, 1)

	#计算损失
	cost = logits * labels #(3, 1)
	cost = 1 - cost #(3, 1)
	cost = tf.reduce_sum(tf.nn.relu(cost)) #()

	#正则化项
	lamda = 1
	l2_loss = lamda * tf.nn.l2_loss(weights) #此公式中已含有1/2
	cost = cost + l2_loss

	#优化器
	learning_rate = 1e-3
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.minimize(cost) #训练过程

	#初始化
	init_op = tf.global_variables_initializer()

	max_step = int(5e4)
	with tf.Session() as sess:
		sess.run(init_op)
		sess.graph.finalize()

		for step in range(max_step): #训练总次数
			_, loss = sess.run([train_op, cost])

			if step % 100 == 0:
				print("Step: %d, loss: %.3f" %(step, loss))


		#保存模型
		weights_result, bias_result = sess.run([weights, bias])

		with open(r'./weights.file', 'wb') as weight_file:
			pickle.dump(weights_result, weight_file)

		with open(r'./bias.file', 'wb') as bias_file:
			pickle.dump(bias_result, bias_file)

	print("Training done for %d steps." %max_step)
	print("weights:", weights_result)
	print("bias:", bias_result)


if __name__ == '__main__':
	tf.reset_default_graph()
	dataset, labelset = loadDataset()
	train_weights(dataset, labelset)
