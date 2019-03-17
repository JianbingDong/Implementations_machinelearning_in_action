"""
此脚本用于实现ｋ均值聚类算法
"""
import numpy as np

Inf = 1e8

def loadData(fileName):
	"""
	用于从文件中读取数据
	#arguments:
		fileName: string. 文件名
	#returns:
		dataMat: list[list[float]]
	"""
	dataMat = []
	with open(fileName, 'r') as file:
		for line in file.readlines():
			curLine = line.strip().split('\t')
			floatLine = list(map(float, curLine)) #将数据转为float型

			dataMat.append(floatLine)

	return dataMat


def distance(vecA, vecB):
	"""
	用于计算向量Ａ、Ｂ之间的距离，此处实现的是欧式距离
	#arguments:
		vecA, vecB:    两个向量
	#returns:
		dist: float, 两个向量之间的距离
	"""

	dist = np.sqrt(np.sum(np.square(vecA - vecB)))
	return dist


def initCentroid(dataMat, k):
	"""
	此函数用于初始化生成随机质心
	#arguments:
		dataMat: np.array, 数据集
		k: integer, 聚类所需的簇数
	#returns:
		centroids: np.array, [k, n] k个随机质心
	"""
	if type(dataMat) != np.ndarray:
		dataMat = np.array(dataMat)

	m, n = dataMat.shape
	centroids = np.zeros((k, n)) #先生成k个质心，每个质心的维度与每个样本的维度相同
	for i in range(n): #遍历每个维度
		minI = np.min(dataMat[:, i]) #所有样本在第ｉ个维度上的最小值
		rangeI = float(np.max(dataMat[:, i]) - minI) #第ｉ个维度的取值范围

		centroids[:, i] = minI + rangeI * np.random.rand(k) #生成k个在 [min, max) 之前的ｉ维度上的值

	return centroids


def kMeans(dataMat, k, disMeas=None, initCent=None):
	"""
	ｋ　means算法的主要计算函数
	#arguments:
		dataMat: np.array, [m, n] 数据集
		k: integer, 聚类簇数
		disMeas: function reference, 指定如何计算向量之间的距离
		initCent: function reference, 制定如何初始化生成质心
	#returns:
		centroids: np.array, [k, n], 计算之后的质心
		clusterAssment: np.array, [m, 2], 每个样本的分配情况，及对应的分配误差
	"""
	if type(dataMat) != np.ndarray:
		dataMat = np.array(dataMat)

	m, n = dataMat.shape
	clusterAssment = np.zeros((m, 2)) #创建一个矩阵来记录每个点簇的分配结果, 第一列记录簇的索引值，第二列记录误差
	centroids = initCent(dataMat, k) #初始生成ｋ个质心

	clusterChanged = True #记录质心是否改变的标志位
	while clusterChanged: #当质心放生了改变
		clusterChanged = False

		for i in range(m): #遍历每个样本
			minDist = Inf #记录最小距离
			minIndex = -1 #最小距离对应的第几个质心

			for j in range(k): #遍历每个质心
				dist_ij = disMeas(centroids[j, :], dataMat[i, :]) #计算第ｉ个样本与第ｊ个质心之间的距离
				if dist_ij < minDist:
					minDist = dist_ij #更新最小距离
					minIndex = j #记录最小质心

			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True

			clusterAssment[i, :] = minIndex, minDist ** 2 #记录第ｉ个样本的质心,　及对应的误差

		# print(centroids)
		#用新分配的簇更新质心
		for cent in range(k):
			ptsInClust = dataMat[np.nonzero(clusterAssment[:, 0] == cent)[0]] #将属于同一个簇内的所有数据点过滤取出
			centroids[cent, :] = np.mean(ptsInClust, axis=0) #计算簇内样本的均值，更新质心

	return centroids, clusterAssment



def show(dataMat, centroids):
	"""
	此函数用于图形化显示
	#arguments:
		dataMat: np.array, [m, n] 数据集
		centroid: np.array, [k, n] 聚类质心
	"""
	import matplotlib.pyplot as plt

	fig = plt.figure()
	subfig = fig.add_subplot(1, 1, 1)

	subfig.scatter(np.transpose(dataMat)[0], np.transpose(dataMat)[1], marker='x', s=25, color='b')


	subfig.scatter(np.transpose(centroids)[0], np.transpose(centroids)[1], marker='o', s=100, color='r')


	plt.show()



if __name__ == '__main__':
	fileName = r'./testSet.txt'
	data = loadData(fileName)

	k = int(input('Please input cluster number: '))

	centroids, clusterAssment = kMeans(data, k=k, disMeas=distance, initCent=initCentroid)
	show(data, centroids=centroids)


