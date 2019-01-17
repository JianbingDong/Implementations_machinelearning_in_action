"""
此函数用于对机器学习实战中的 SMO简易算法进行实现
@author: Jianbing Dong
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
from MySvmNeuralNetwork import loadDataset

def loadData(filename):
	"""
	用于生成数据集
	#arguments:
		filename: string, txt文件名
	#returns:
		dataMat: list, like [[1.2, 2.4],
							 [2.1, 3.4],...] 每行代表一个样本，每列代表一个特征
		labelMat: list, like [1, -1, 1, 1, -1, ...]
	"""	
	dataMat = []
	labelMat = []
	with open(filename, 'r') as file:
		for line in file.readlines():
			lineArr = line.strip().split('\t')
			dataMat.append([float(lineArr[0]), float(lineArr[1])])
			labelMat.append(float(lineArr[-1]))
	return dataMat, labelMat


def showDataset(dataMat, labelMat):
	"""
	用于在图像中刻画数据集
	dataMat, labelMat是loadData函数的返回值
	"""
	fig = plt.figure()
	plt.title('Dataset')
	subfig = fig.add_subplot(1, 1, 1)
	subfig.set_xlabel('x1')
	subfig.set_ylabel('x2')

	for i in range(len(dataMat)):
		data = dataMat[i]
		label = labelMat[i]
		if -1 == label:
			subfig.scatter(data[0], data[1], marker='x', s=50)
		else:
			subfig.scatter(data[0], data[1], marker='o', s=50)

	plt.show()


def selectJrand(i, m):
	"""
	用于选择一个不等于i的随机数，随机数的取值从[0-m]
	"""
	j = i
	while j==i:
		j = np.random.randint(m, size=1) #m不能被取到

	return j

def clipAlpha(aj, H, L):
	"""
	用于对aj进行范围裁剪，根据《统计学习方法》的公式7.108
	"""
	if aj > H:
		aj = H
	if aj < L:
		aj = L
	return aj


def SMO_simple(dataMat, labelMat, C, epsilon, maxStep):
	"""
	SMO算法的简易实现版本 SMO: Sequential Minimal Optimization，依照《机器学习实战》
	#arguments:
		dataMat, labelMat: loadData()的返回值
		C: 惩罚参数，是 使间隔尽量大 与 误分类点尽量少 的调和系数，C>0
		epsilon: epsilon检验范围
		maxStep: integer, 最大循环次数
	#returns:
		alphas: 参数
		b: bias
	"""
	dataMatrix = np.mat(dataMat) #(100, 2)
	labelMatrix = np.mat(labelMat).transpose() #(100, 1)

	b = 0 #分割超平面的偏置
	m, n = dataMatrix.shape #m为样本数量, n为特征数量
	alphas = np.zeros((m, 1)) # 创建(100, 1)个alpha变量，并初始化为0
	iter_ = 0 #记录迭代次数

	while iter_ < maxStep:
		alphaPairsChanged = 0 #用于记录更改了几对alpha，在遍历一次数据集的过程中
		for i in range(m): #遍历每个样本
			gx_i = float(np.sum(np.multiply(np.multiply(alphas, labelMatrix),
				np.matmul(dataMatrix, dataMatrix[i].transpose())))) + b #《统计学习方法》公式7.104
			E_i = gx_i - float(labelMatrix[i]) #公式7.105

			if (alphas[i] > 0 and labelMatrix[i] * E_i > epsilon) or \
				(alphas[i] < C and labelMatrix[i] * E_i < -epsilon): #检验支持向量是否违反KKT条件，公式7.112，若违反
				j = selectJrand(i, m) #从其余alpha中随机选取一个
				gx_j = gx_i = float(np.sum(np.multiply(np.multiply(alphas, labelMatrix),
				np.matmul(dataMatrix, dataMatrix[j].transpose())))) + b #用同样的方法计算gx_j
				E_j = gx_j - float(labelMatrix[j]) 

				#先记录未进行优化之前的alpha值
				alpha_i_old = alphas[i].copy() 
				alpha_j_old = alphas[j].copy()

				#根据y的不同取值，计算上界和下界，《统计学习方法》P.126左下角
				if labelMatrix[i] != labelMatrix[j]: #二者不相等
					L = np.maximum(0, alpha_j_old - alpha_i_old)
					H = np.minimum(C, C + alpha_j_old - alpha_i_old)
				else:
					L = np.maximum(0, alpha_j_old + alpha_i_old - C)
					H = np.minimum(C, alpha_j_old + alpha_i_old)

				if L == H: #若上、下界相等，则不需要优化，因为此时的alpha_i和alpha_j已经在《统计学习方法》图7.8的角落上
					print("L==H: %s" %L)
					continue

				eta = dataMatrix[i] * dataMatrix[i].transpose() +\
					  dataMatrix[j] * dataMatrix[j].transpose() -\
					  2 * dataMatrix[i] * dataMatrix[j].transpose() #按照公式7.107计算eta
				if eta <= 0:
					print("eta <= 0")
					continue

				#更新alpha_j
				alphas[j] = alpha_j_old + (labelMatrix[j] * (E_i - E_j)) / eta #公式7.106
				#裁剪alpha_j的范围
				alphas[j] = clipAlpha(alphas[j], H, L) #公式7.108

				if np.abs(alphas[j] - alpha_j_old) < 1e-8:
					print("J not moving enough") #alpha_j 更新不大
					continue

				#更新alpha_i
				alphas[i] = alpha_i_old + labelMatrix[i] * labelMatrix[j] * (alpha_j_old - alphas[j]) #公式7.109

				#更新b，公式7.115和7.116
				b1_new = b - E_i - labelMatrix[i] * dataMatrix[i] * dataMatrix[i].transpose() *\
						 (alphas[i] - alpha_i_old) - labelMatrix[j] * dataMatrix[j] * dataMatrix[i].transpose() *\
						 (alphas[j] - alpha_j_old)

				b2_new = b - E_j - labelMatrix[i] * dataMatrix[i] * dataMatrix[j].transpose() *\
						 (alphas[i] - alpha_i_old) - labelMatrix[j] * dataMatrix[j] * dataMatrix[j].transpose() *\
						 (alphas[j] - alpha_j_old)

				if 0 < alphas[i] and alphas[i] < C:
					b = b1_new
				elif 0 < alphas[j] and alphas[j] < C:
					b = b2_new
				else:
					b = (b1_new + b2_new) / 2

				alphaPairsChanged += 1 #更改了一对alpha
				print("Iteration: %d, i: %d, pairs changed %d" %(iter_, i, alphaPairsChanged))

		if alphaPairsChanged == 0: #若本次遍历数据集未对alpha进行优化
			iter_ += 1
		else:
			iter_ = 0
		print("Iteration number %d" %iter_) #一共已有多少次未对alpha进行优化

	result = {'alphas': alphas, 'bias': b}
	with open(r'./result.file', 'wb') as file:
		pickle.dump(result, file)

	return alphas, b


def showDataset_support_vector(dataMat, labelMat, alphas, b):
	"""
	用于在图像中刻画数据集
	dataMat, labelMat是loadData函数的返回值
	alphas, b为SMO_simple()函数的返回值
	并标注支持向量
	"""
	fig = plt.figure()
	plt.title('Dataset')
	subfig = fig.add_subplot(1, 1, 1)
	subfig.set_xlabel('x1')
	subfig.set_ylabel('x2')
	subfig.axis([-2, 12, -8, 6])

	dataMatrix = np.mat(dataMat)
	labelMatrix = np.mat(labelMat).transpose()

	for i in range(len(dataMat)):
		data = dataMat[i]
		label = labelMat[i]
		if -1 == label:
			subfig.scatter(data[0], data[1], marker='s', s=50, color='b')
			if alphas[i] > 0: #是支持向量
				subfig.scatter(data[0], data[1], marker='o', edgecolor='r', s=200, color='')
		else:
			subfig.scatter(data[0], data[1], marker='d', s=50, color='y')
			if alphas[i] > 0: #是支持向量
				subfig.scatter(data[0], data[1], marker='o', edgecolor='r', s=200, color='')
				

	weights = np.sum(np.multiply(np.multiply(alphas, labelMatrix), dataMatrix), axis=0)
	show_x1 = np.linspace(0, 8, 100)
	show_x2 = np.array(-(b + weights[0, 0] * show_x1) / weights[0, 1])
	show_x2 = np.reshape(show_x2, newshape=(show_x2.shape[-1], ))
	print(show_x2.shape)
	print(show_x1.shape)


	subfig.plot(show_x1, show_x2, '-', color='g')

	plt.show()



if __name__ == '__main__':
	training = True #是否需要进行训练
	filename = r'./testSet.txt'
	# dataMat, labelMat = loadData(filename) #《机器学习实战》的样本
	dataMat, labelMat = loadDataset() #《统计学习方法》例7.1的样本
	# showDataset(dataMat, labelMat) #显示数据集
	if os.path.exists(r'./result.file') and not training:
		print('loading...')
		with open(r'./result.file', 'rb') as file:
			result = pickle.load(file)
			alphas = result['alphas']
			b = result['bias']
	else:		
		alphas, b = SMO_simple(dataMat, labelMat, C=0.6, epsilon=1e-3, maxStep=40)


	print(alphas[alphas > 0])
	print(b)

	showDataset_support_vector(dataMat, labelMat, alphas, b)
