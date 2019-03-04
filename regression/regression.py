"""
此脚本用于测试回归型机器学习方法
"""

import numpy as np
from matplotlib import pyplot as plt

"""
ordinary least squares, 普通最小二乘法
"""
def loadData(filename):
	"""
	用于从文件中载入数据
	#arguments:
		filename: string, where to find this file.
	#returns:
		dataMat: two dimension data design matrix.
		labels: list
	"""
	numFeat = len(open(filename).readline().split('\t')) - 1 #通过读出一行的数据个数，来获取一个样本的数据的维度个数
	dataMat = []
	labels = []
	with open(filename, 'r') as file:
		for line in file.readlines():
			lineArr = [] #用于记录每一行的数据
			curLine = line.strip().split('\t') #将当前行分隔为list
			for i in range(numFeat): #对于样本数据的每个维度
				lineArr.append(float(curLine[i])) #将其加入到数据记录list中
			dataMat.append(lineArr)
			labels.append(float(curLine[-1])) #记录label

	return dataMat, labels


def standRagres(xArr, yArr):
	"""
	用于计算标准最小二乘法
	#arguments:
		xArr: 输入数据设计矩阵;
		yArr: 标签，list
	"""
	xMat = np.mat(xArr) #shape = [m, n]

	yMat = np.mat(yArr).T #shape = [m, 1]

	xTx = xMat.T * xMat #xT * x, shape = [n, n]

	if np.linalg.det(xTx) == 0.0: #判断矩阵是否可逆,可逆＝非奇异＝行列式不为0
		print("THis matrix is singular, connot do inverse")
		return

	w = xTx.I * xMat.T * yMat #计算最小二乘估计结果

	return w


def showResult(x, y, w):
	"""
	此函数用于显示原始数据以及拟合出来的直线
	#arguments:
		x: 二维数据设计矩阵
		y: list
		w: the regresion coefficients
	"""
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	x = np.array(x)
	ax.scatter(x[:, 1], y) #以散点图画原始数据

	xCopy = np.copy(x)
	yCopy = np.copy(y)

	yHat = xCopy * w

	ax.plot(xCopy[:, 1], yHat) #画出拟合直线

	plt.show()


"""
局部加权线性回归, locally weighted linear regression, LWLR
每次预测新样本时，都会重新计算一遍ws
"""

def lwlr(testPoint, xArr, yArr, k=1.0):
	"""
	通过局部加权线性回归的方法求出回归系数.
	#arguments:
		testPoint: 测试样本点，一个样本
		xArr:　所有训练样本
		yArr: 所有训练标签
		k:　超参数，控制权重衰减的速度
	#returns:
		ws: 回归系数
	"""
	xMat = np.mat(xArr) #shape = [m, n]
	yMat = np.mat(yArr).T #shape = [m, 1]
	testPoint = np.mat(testPoint) #shape = [1, n]

	m = xMat.shape[0] #样本个数
	weights = np.mat(np.eye(m)) #创建对角矩阵, 数据权重矩阵，非回归系数, 为每个样本点都创建了一个权重
	for j in range(m): #遍历数据集
		diffMat = testPoint - xMat[j, :] #计算测试样本点和第j个样本点的差
		weights[j, j] = np.exp(diffMat * diffMat.T / (-2 * k** 2)) #计算高斯核的值

	xTx = xMat.T * (weights * xMat)
	if np.linalg.det(xTx) == 0.0:
		print("This matrix is singular, cannot do inverse.")
		return
	ws = xTx.I * (xMat.T * (weights * yMat))

	return ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
	"""
	用于测试局部加权线性回归
	#arguments:
		testArr: 测试样本
		xArr:　所有训练样本
		yArr:　所有训练标签
		k: 超参数，控制样本权重衰减的速度
	#returns:
		yHat: 每个测试样本点的预测值
	"""
	if not isinstance(testArr, np.ndarray):
		testArr = np.array(testArr)
		xArr = np.array(xArr)
		yArr = np.array(yArr)
	else:
		pass

	m = testArr.shape[0]
	yHat = np.zeros((m, ))
	for i in range(m):
		yHat[i] = testArr[i] * lwlr(testArr[i], xArr, yArr, k)

	return yHat


def showLwlr(xArr, yArr, yHat):
	"""
	用于显示lwlr的拟合结果
	#arguments:
		xArr: 数据设计矩阵
		yArr: 标签
		yHat:　预测值
	"""
	xMat = np.mat(xArr)
	srtIndex = xMat[:, 1].argsort(0)
	xSort = xMat[srtIndex][:, 0, :]

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(xSort[:, 1], yHat[srtIndex])

	ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red')

	plt.show()




if __name__ == '__main__':
	filename = r'./ex0.txt'
	dataMat, labels = loadData(filename)


	# w = standRagres(dataMat, labels)
	# if w is not None:
		# print(w)
		# showResult(dataMat, labels, w)

	# print(dataMat[0] * lwlr(dataMat[0], dataMat, labels, k=1.0))
	yHat = lwlrTest(dataMat, dataMat, labels, k=0.01)
	showLwlr(dataMat, labels, yHat)




