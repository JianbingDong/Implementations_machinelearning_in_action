"""
此脚本用于实现　回归问题的提升树算法，以及梯度提升决策树　(Gradient Boosting Decision Tree)
"""

import numpy as np
from AdaBoost import Inf
import random

def splitData(dataArray, dimension, threshValue):
	"""
	根据阈值对数据集进行划分
	#arguments:
		dataArray: 数据设计矩阵;
		dimension: integer, 使用哪个维度进行计算
		threshValue: float, 当前维度上所选择的切分阈值
	#returns:
		splitIndex: list, 行向量，代表每个样本属于某个集合，０属于R1, 1属于R2
	"""
	if isinstance(dataArray, np.ndarray):
		pass
	else:
		dataArray = np.array(dataArray)

	m = dataArray.shape[0] #样本个数

	splitIndex = np.ones((m, 1), dtype=np.int32) #切分索引向量，用于记录各样本属于哪个集合，０属于R1集合，１属于R2集合

	splitIndex[dataArray[:, dimension] <= threshValue] = 0 #小于阈值的则分到R1集合

	splitIndex = np.reshape(splitIndex, (splitIndex.shape[0], )) #变为行向量

	return splitIndex



def regressionBuildStump(dataArray, labels):
	"""
	此函数用于构造回归问题的决策树桩. 找到最佳决策树
	#arguments:
		dataArray: 数据设计矩阵;
		labels: 样本标签值
	"""
	if isinstance(dataArray, np.ndarray):
		pass
	else:
		dataArray = np.array(dataArray)

	m, n = dataArray.shape #样本数量，样本维度

	numSteps = 10 #线搜索阈值时的步数
	bestStump = {} #当前最佳决策树

	minError = Inf #记录最小的值，即当前维度，当前切分点下的最小误差,式子(5.21)

	for i in range(n): #遍历每个特征维度
		rangeMin = np.min(dataArray[:, i]) #当前维度上的最小值
		rangeMax = np.max(dataArray[:, i]) #当前维度上的最大值

		stepSize = (rangeMax - rangeMin) / numSteps #计算当前维度上的阈值搜索步长

		# for j in range(-1, int(numSteps) + 1): #遍历所有阈值
		for j in range(1, 10):

			# threshValue = rangeMin + float(j) * stepSize #计算当前阈值
			threshValue = float(j) + 0.5

			splitIndex = splitData(dataArray, i, threshValue) #按阈值切分数据集

			y1 = np.multiply(labels, (1 - splitIndex)) #找出属于Ｒ１的样本标签值
			y2 = np.multiply(labels, splitIndex) #找出属于R2的样本标签值

			if np.sum(1 - splitIndex) == 0:
				c1 = 0
			else:
				c1 = np.sum(y1) / np.sum(1 - splitIndex) #用式(5.20)计算c1
			if np.sum(splitIndex) == 0:
				c2 = 0
			else:
				c2 = np.sum(y2) / np.sum(splitIndex) #计算c2

			splitError = np.sum(np.square(y1 - c1) * (1 - splitIndex)) +\
						 np.sum(np.square(y2 - c2) * splitIndex) #计算当前阈值对应的切分误差

			if splitError < minError:
				minError = splitError
				bestStump['dimension'] = i #当前切分维度
				bestStump['threshValue'] = threshValue #当前切分阈值
				bestStump['c1'] = c1 #小于阈值的输出值
				bestStump['c2'] = c2 #大于阈值的输出值

	return bestStump


def calWithBoostingTree(weakStumps, inData):
	"""
	计算提升树的输出
	#arguments:
		weakStumps: list, 弱分类器列表
		inData: 待计算数据的设计矩阵, (m, n)
	#returns:
		result: 行向量
	"""
	if isinstance(inData, np.ndarray):
		pass
	else:
		inData = np.array(inData)

	m, n = inData.shape

	result = np.zeros((m, 1))
	for stump in weakStumps:
		result[inData[:, stump['dimension']] < stump['threshValue']] += stump['c1']
		result[inData[:, stump['dimension']] >= stump['threshValue']] += stump['c2']

	result = np.reshape(result, (m, )) #变为行向量

	return result


def regressionAdaBoostTrain(dataArray, labels, stopResidue, numIterations=40):
	"""
	回归型提升树的训练
	#arguments:
		dataArray: 数据的设计矩阵;
		labels: list，样本数据的标签值
		stopResidue: float, 残差小于该值时则停止循环
		numIterations: integer, 最大循环次数
	"""
	weakStumps = [] #保存弱分类器
	originalLabel = np.copy(labels) #记录原始标签，用于计算提升树的残差

	if isinstance(dataArray, np.ndarray):
		pass
	else:
		dataArray = np.array(dataArray)

	for iteration in range(numIterations):
		bestStump = regressionBuildStump(dataArray, labels) #求出当前决策树桩

		weakStumps.append(bestStump) #记录该树桩

		results = calWithBoostingTree(weakStumps, dataArray) #计算当前提升树预测结果

		residues = originalLabel - results #计算训练数据的残差
		print("Current residues: ", np.sum(np.square(residues)))

		if np.sum(np.square(residues)) < stopResidue: #满足停止条件
			break

		else:
			labels = residues #否则将标签值更新为上一步的残差，继续循环

	return weakStumps


if __name__ == '__main__':
	dataMat = range(1, 11)
	dataMat = np.reshape(dataMat, (len(dataMat), 1))
	labels = [5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]

	stumps = regressionAdaBoostTrain(dataMat, labels, stopResidue=0.1, numIterations=40)
	for stump in stumps:
		print(stump['threshValue'])
