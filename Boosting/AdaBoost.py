"""
This script is used to test the AdaBoost algorithm.
"""
import numpy as np
from matplotlib import pyplot as plt

Inf = 1e16

def loadSimpData():
	"""
	For creating simple data sets.
	"""
	dataMat = [[1., 2.1],
			   # [1.5, 1.6],
			   [2, 1.1],
			   [1.3, 1.],
			   [1., 1.],
			   [2., 1.]]

	labels = [1.0, 1.0, -1.0, -1.0, 1.0]

	return dataMat, labels

def showDataset(dataMat, labels):
	"""
	This function is used to show the data sets.
	#arguments:
		dataMat: two-dimensional list, the design matrix;
		labels: list, the label of each data.
	"""
	fig = plt.figure()
	plt.title('Dataset')
	subfig = fig.add_subplot(1, 1, 1)
	subfig.set_xlabel('x1')
	subfig.set_ylabel('x2')

	for i in range(len(dataMat)):
		data = dataMat[i]
		label = labels[i]
		if -1 == int(label):
			subfig.scatter(data[0], data[1], marker='x', s=50)
		else:
			subfig.scatter(data[0], data[1], marker='o', s=50)

	plt.show()



def stumpClassify(dataMat, dimension, threshValue, threshIneq):
	"""
	对数据进行阈值比较，在阈值某一侧，则置为1，另一侧置为-1。
	#arguments:
		dataMat: 数据设计矩阵;
		dimension: integer, 使用样本哪个维度的数据用于与阈值比较；
		threshValue: float, 指定的阈值大小
		threshIneq:　比较的方式: 小于(less than)　lt; 大于(great than)　gt
	#returns:
		result: array; 记录了与阈值比较之后的结果，不同位置对应不同样本的判断结果
	"""
	if isinstance(dataMat, np.ndarray):
		pass
	else:
		dataMat = np.array(dataArray)

	result = np.ones((dataMat.shape[0], 1)) #构建一个(m, 1)的全１矩阵
	if threshIneq == 'lt':
		result[dataMat[:, dimension] <= threshValue] = -1.0
	else:
		result[dataMat[:, dimension] > threshValue] = -1.0

	return result


def buildStump(dataArray, labels, D):
	"""
	找到数据集上“最佳”的单层决策树，即决策树桩decision stump.
	#arguments:
		dataArray: 数据设计矩阵；
		labels: 标签向量;
		D:　权重向量
	"""
	if isinstance(dataArray, np.ndarray):
		dataMat = dataArray
	else:
		dataMat = np.array(dataArray)
	labelMat = np.transpose(np.array(labels))

	m,n = dataMat.shape

	##进行线搜索，在最大、最小值区间内选取阈值
	numSteps = 10.0
	bestStumps = {}
	bestClasEst = np.zeros((m, 1))
	minError = Inf

	for i in range(n): #遍历每个样本的n个维度，当前维度ｉ
		rangeMin = np.min(dataMat[:, i]) #找到i 维度上的最大值和最小值
		rangeMax = np.max(dataMat[:, i])

		stepSize = (rangeMax - rangeMin) / numSteps #计算出每一步阈值步进的步长

		for j in range(-1, int(numSteps) + 1): #遍历所有可能取到的阈值
			for inEqual in ['lt', 'gt']: #在大于和小于两种方式间遍历 
				threshValue = (rangeMin + float(j) * stepSize) #计算当前的阈值
				predictedValues = stumpClassify(dataMat, i, 
									threshValue, inEqual) #带入函数计算数据集在当前阈值下的分类结果
				
				errArray = np.ones((m, 1)) #构建一个全1矩阵，用于存储每个样本是否被误分类

				errArray[predictedValues == np.reshape(labelMat, (len(labelMat), 1))] = 0 #将未被误分类的样本对应位置置为０

				weightedError = np.sum(D * errArray) #所有样本的加权误差

				if weightedError < minError:
					minError = weightedError
					bestClasEst = predictedValues.copy() #所有样本的预测类别
					bestStumps['dim'] = i #记录当前所用的特征维度
					bestStumps['thresh'] = threshValue #当前所使用的阈值
					bestStumps['ineq'] = inEqual #当前所使用的阈值比较方式

	return bestStumps, minError, bestClasEst


def adaBoostTrainDecisionTree(dataArr, labels, numIteration=40):
	"""
	Adaboost　训练过程，由于弱分类器是决策树，所以也可以叫做　提升数。
	#arguments:
		dataArr: 数据设计矩阵;
		labels: 标签;
		numIteration: integer, 最大循环次数
	#returns:
		no returns.
	"""
	weakClassArr = [] #用于记录弱分类器
	dataArr = np.array(dataArr)

	m = dataArr.shape[0] #样本个数
	D = np.ones((m, 1)) / m #初始样本权重

	aggClassEst = np.zeros((m, 1))
	for i in range(numIteration): #开始遍历numIteration次

		##只要将下面这个构建决策树的弱分类器方式更换为其他弱分类器，则可以实现其他方式的AdaBoost
		bestStumps, error, bestClasEst = buildStump(dataArr, labels, D) #以当前权重向量Ｄ，生成一个决策树桩


		print("Current D: ", np.transpose(D)) #输出当前权重向量

		alpha = float(0.5 * np.log((1 - error) / np.maximum(error, 1e-16) + 1e-16)) #用式8.2计算当前分类器的权重alpha

		bestStumps['alpha'] = alpha #在决策树中记录此信息
		weakClassArr.append(bestStumps) #加入到弱分类器列表
		print("Current estimated class: ", np.transpose(bestClasEst)) #输出当前样本预测结果

		expAlpha = np.multiply(-1 * alpha * np.reshape(labels, (len(labels), 1)), bestClasEst) #计算所有 -alpha_m * yi * Gm(xi)

		D_new = np.multiply(D, np.exp(expAlpha)) #计算所有wmi * exp(-alpha_m * yi * Gm(xi)),　即式8.4的分子
		D = D_new / np.sum(D_new) #将其归一化, 即式8.4的分母

		aggClassEst += alpha * bestClasEst #构建分类器线性组合, 即将当前数据带入到式子8.6的计算结果
		print("Current aggregate class estimation: ", np.transpose(aggClassEst))

		aggErrors = np.multiply(np.sign(aggClassEst) != np.reshape(labels, (len(labels), 1)), np.ones((m, 1))) #线性组合分类器的错误分类样本

		errorRate = np.sum(aggErrors) / m #错误分类样本率
		print("Curent total error: ", errorRate)

		print('\n')
		if errorRate == 0.0:
			break

	return weakClassArr








if __name__ == '__main__':
	# dataMat, labels = loadSimpData()
	# showDataset(dataMat, labels)
	dataMat = range(10)
	dataMat = np.reshape(dataMat, (len(dataMat), 1))
	labels = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]

	weakClassArr = adaBoostTrainDecisionTree(dataMat, labels, numIteration=40)
	for item in weakClassArr:
		print(item['alpha'])
