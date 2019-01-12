"""
此脚本用于构建logistic回归的基本函数
@author: Jianbing Dong
"""

import numpy as np
import random

def loadDataset():
	"""
	此函数用与构造数据集
	dataMat: list, [[1, x1, x2],
					[1, x1, x2],
					...]shape为[100, 3]，100个样本，每个样本有3个特征
	labelMat: list, [0, 1, 0, 1, ...]
	"""
	dataMat = []
	labelMat = []
	with open('./testSet.txt') as file:# 此文件中，前两列为每个样本的特征，最后一列为该样本的标签，每一行代表不同的样本
		for line in file.readlines():
			lineArr = line.strip().split() #先移除这一行内容头尾的空格和换行符，然后按空格将其分开，返回list
			dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #将特征添加到dataMat中
			labelMat.append(int(lineArr[2]))

	return dataMat, labelMat

def sigmoid(inX):
	"""
	此函数用于计算sigmoid方程的输出
	"""
	return 1.0 / (1 + np.exp(-inX))


def gradientAscent(dataMatin, classLabels):
	"""
	此函数使用梯度上升算法对回归系数进行优化求解。
	此版本是使用全部训练数据进行迭代优化。
	#arguments:
		dataMatin: loadDataset函数的返回结果
		classLabels: loadDataset函数的返回结果
	#returns:

	"""
	dataMatrix = np.mat(dataMatin) #将list转为array，（100, 3）
	labelMatrix = np.mat(classLabels).transpose() #将list转为array，并转置，最终shape为(100, 1)

	m, n = dataMatrix.shape #获取样本个数m，及每个样本特征数n

	alpha = 1e-3 #学习率
	maxCycles = 500 #最大迭代次数
	weights = np.ones((n, 1)) #将回归系数初始化为1，(n, 1)
	for k in range(maxCycles):
		#使用全部数据进行一次计算
		h = sigmoid(dataMatrix * weights) #矩阵乘法，h.shape为(100, 1)
		error = (labelMatrix - h) #极大似然估计的化简推导结果
		weights = weights + alpha * dataMatrix.transpose() * error #极大似然估计的化简推导结果
	
	return weights


def gradientAscent_batch(dataMatin, classLabels, batch=None):
	"""
	此函数使用梯度上升算法对回归系数进行优化求解。
	此版本是使用batch_size个训练样本进行mini_batch的训练。
	#arguments:
		dataMatin: loadDataset函数的返回结果
		classLabels: loadDataset函数的返回结果
		batch: 用于指定每次选用多少训练数据进行计算，默认为None，代表使用全部训练数据
	#returns:

	"""
	if batch is None:
		batch = len(dataMatin)

	dataMatrix = np.mat(dataMatin) #将list转为array，（100, 3）
	labelMatrix = np.mat(classLabels).transpose() #将list转为array，并转置，最终shape为(100, 1)

	_, n = dataMatrix.shape #获取样本个数m，及每个样本特征数n

	alpha = 1e-3 #学习率
	maxCycles = 500 #最大迭代次数
	weights = np.ones((n, 1)) #将回归系数初始化为1，(n, 1)

	for k in range(maxCycles):
		index_ = list(range(len(dataMatin)))
		random.shuffle(index_)
		for i in range(0, len(index_), batch):
			end = i + batch
			if end > len(index_): 
				end = len(index_)
			sample_index = index_[i: end]

			dataBatch = dataMatrix[sample_index]
			labelBatch = labelMatrix[sample_index]

			#每次使用batch个数据进行一次计算
			h = sigmoid(dataBatch * weights) #矩阵乘法，h.shape为(100, 1)
			error = (labelBatch - h) #极大似然估计的化简推导结果
			weights = weights + alpha * dataBatch.transpose() * error #极大似然估计的化简推导结果
	
	return weights


def getBatch(dataMatrix, labelMatrix, batch, maxStep):
	"""
	用函数生成器的方式来实现获取batch，算法效率没有被提升，
	只是用一个函数将取batch的方式封装起来了，在调用时显得更简洁。
	"""
	step = 0
	while step < maxStep:
		index_ = list(range(dataMatrix.shape[0]))
		random.shuffle(index_)
		for i in range(0, len(index_), batch):
			end = i + batch
			if end > len(index_): 
				end = len(index_)
			sample_index = index_[i: end]

			dataBatch = dataMatrix[sample_index]
			labelBatch = labelMatrix[sample_index]	
			
			yield dataBatch, labelBatch	

		step += 1

def gradientAscent_batch_2(dataMatin, classLabels, batch=None):
	"""
	此函数使用梯度上升算法对回归系数进行优化求解。
	此版本是使用batch_size个训练样本进行mini_batch的训练。
	此版本内部使用函数生成器来获取batch
	#arguments:
		dataMatin: loadDataset函数的返回结果
		classLabels: loadDataset函数的返回结果
		batch: 用于指定每次选用多少训练数据进行计算，默认为None，代表使用全部训练数据
	#returns:

	"""
	if batch is None:
		batch = len(dataMatin)

	dataMatrix = np.mat(dataMatin) #将list转为array，（100, 3）
	labelMatrix = np.mat(classLabels).transpose() #将list转为array，并转置，最终shape为(100, 1)

	_, n = dataMatrix.shape #获取样本个数m，及每个样本特征数n

	alpha = 1e-3 #学习率
	maxCycles = 500 #最大迭代次数
	weights = np.ones((n, 1)) #将回归系数初始化为1，(n, 1)

	for dataBatch, labelBatch in getBatch(dataMatrix, labelMatrix, batch, maxStep=maxCycles):
		#每次使用batch个数据进行一次计算
		h = sigmoid(dataBatch * weights) #矩阵乘法，h.shape为(100, 1)
		error = (labelBatch - h) #极大似然估计的化简推导结果
		weights = weights + alpha * dataBatch.transpose() * error #极大似然估计的化简推导结果

	return weights	


if __name__ == '__main__':
	dataMat, labelMat = loadDataset()
	weights = gradientAscent_batch(dataMat, labelMat, batch=20)
	# weights = gradientAscent_batch_2(dataMat, labelMat, batch=100)
	print(weights)
