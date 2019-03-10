"""
此脚本用于编写机器学习实战　第９章　树回归。
"""

import numpy as np
Inf = 1e8

def loadData(fileName):
	"""
	从文件中读取数据
	#arguments:
		fileName: string, where to find the data.
	#returns:
		dataMat: list of lists. [[1.1, 1.2, ...], ...]，　数据的最后一个维度为对应样本的标签值
	"""
	dataMat = []
	with open(fileName, 'r') as file:
		for line in file.readlines():
			curLine = line.strip().split('\t')
			floatLine = list(map(float, curLine))
			dataMat.append(floatLine)

	return dataMat


def binSplitData(dataSet, feature, value):
	"""
	对数据集进行二分。
	#arguments:
		dataSet: np.array with shape [numSamples, featuersDim]
		feature: integer, which dimension of feature used to split the dataset.
		value: float, the correspoding feature value.
	#returns:

	"""
	mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]#[0] #nonzero返回的是数组里面非零元素的索引值，每个非零元素对应一个tuple??最后为和还要取第０维
	mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]

	return mat0, mat1

def createTree(dataSet, leafType=None, errType=None, ops=(1, 4)):
	"""
	用于创建CART树
	#arguments:
		dataSet: np.array with shape [numSamples, featureDim]
		leafType: 如何建立叶节点的函数；
		errType: 计算误差的函数;
		ops: 其他可选参数
	#returns:
		返回创建好的一颗树
	"""
	feature, value = chooseBestSplit(dataSet, leafType, errType, ops) #查找当前数据集下的最佳切分维度及对应的切分点
	if feature is None: #不需要进行切分，返回叶节点
		return value

	retTree = {} #基于当前数据集创建树的根节点
	retTree['splitIndex'] = feature #记录切分特征维度
	retTree['splitValue'] = value #记录切分值
	leftSet, rightSet = binSplitData(dataSet, feature, value) #对当前数据集进行二分
	retTree['leftNode'] = createTree(leftSet, leafType, errType, ops) #创建左子树
	retTree['rightNode'] = createTree(rightSet, leafType, errType, ops) #创建右子树

	return retTree


def regLeaf(dataset):
	"""
	用于回归树生成叶节点
	#dataset: np.array with shape [numSamples, featureDim]
	"""
	return np.mean(dataset[:, -1]) #计算当前数据集上的标签值的均值

def regErr(dataset):
	"""
	用于回归树计算数据集误差
	#dataset: np.array with shape [numSamples, featureDim]
	"""
	return np.var(dataset[:, -1]) * dataset.shape[0] #计算当前数据集上的标签的误差

def chooseBestSplit(dataset, leafType=None, errType=None, ops=(1, 4)):
	"""
	基于当前数据集，选择出最佳切分点
	#arguments:
		dataset: np.array with shape [numSamples, featureDim]
		leafType: the reference of the function to build leaf node.
		errType: the reference of the function to calculate the error in this dataset.
		ops: 其他参数, 在此例子中，第一个参数为tols: 允许误差下降的值，第二个为tolN: 切分的最少的样本数, 即预剪枝技术
	#returns:
		index, splitvalue: 切分维度和所用的切分值
	"""
	tols, tolN = ops

	if len(set(dataset[:, -1].transpose().tolist())) == 1: #所有样本的标签值都相同
		return None, leafType(dataset) #不进行切分，直接生成叶节点

	m, n = dataset.shape
	split = errType(dataset) #不进行切时的误差

	bestSplit = Inf #记录切分误差
	bestIndex = 0 #记录切分的维度
	bestValue = 0 #记录切分的值

	for featureIndex in range(n - 1): #遍历从[0, n-1]上的维度，因为最后一个维度是对应样本的标签
		for splitValue in set(dataset[:, featureIndex]): #在当前维度上的所有特征取值来作为可能的切分点，也可用其他方式生成可能的切分点
			mat0, mat1 = binSplitData(dataset, featureIndex, splitValue) #根据切分值与切分维度，将当前数据集二分
			if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN): 
				continue #切分之后的数据集中的样本个数不满足要求

			newSplit = errType(mat0) + errType(mat1) #计算当前切分情况下的数据集误差
			if newSplit < bestSplit:
				bestIndex = featureIndex
				bestValue = splitValue
				bestSplit = newSplit

	if (split - bestSplit) < tols: #若未切分的数据集误差比切分后的数据集误差小
		return None, leafType(dataset) #则没必要进行切分

	mat0, mat1 = binSplitData(dataset, bestIndex, bestValue) #用最佳维度，最佳且分点进行切分
	if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN): #切分之后的数据集中的样本个数不满足要求
		return None, leafType(dataset) 

	return bestIndex, bestValue


#用于后剪枝
def isTree(obj):
	"""
	用于判断一个对象是否是一棵树，由于树使用dict类型表示的，所以一棵树的类型为dict
	#argument:
		obj: 待判断的对象
	#returns:
		True or False: 代表是否是一棵树
	"""
	return (type(obj).__name__ == 'dict')


def getMean(tree):
	"""
	获得整颗树的叶节点平均值，会改变原来生成的树的结构,即合并所有节点，返回一个根节点，输出值为合并节点的均值
	"""
	if isTree(tree['rightNode']):
		tree['rightNode'] = getMean(tree['rightNode'])
	if isTree(tree['leftNode']):
		tree['leftNode'] = getMean(tree['leftNode'])

	return (tree['leftNode'] + tree['rightNode']) / 2.0

def postPruning(tree, testData):
	"""
	用于对树进行后剪枝, 递归地调用此函数，先从根节点访问到叶节点，然后逐步从叶节点开始决定是否进行合并(即剪枝)
	#arguments:
		tree: dict, 之前创建出来的树
		testData: np.array with shape [numSamples, featureDim],　最后一个维度为样本的标签值
	#returns:

	"""
	if testData.shape[0] == 0: #若测试数据为空，则返回整颗树的均值，即将整棵树合并只剩下一个根节点，输出为所有叶节点合并之后的均值
		return getMean(tree)

	if (isTree(tree['rightNode']) or isTree(tree['leftNode'])): #若当前树至少有一个子节点是一棵树, 将测试数据集依照当前树的切分点，进行二分
		lset, rset = binSplitData(testData, tree['splitIndex'], tree['splitValue'])

	if isTree(tree['leftNode']): #当前树的左节点为一棵树
		tree['leftNode'] = postPruning(tree['leftNode'], lset) #对左节点利用切分之后的数据集进行剪枝
	if isTree(tree['rightNode']): #当前树的右节点为一棵树
		tree['rightNode'] = postPruning(tree['rightNode'], rset)

	if not isTree(tree['leftNode']) and not isTree(tree['rightNode']): #左右节点都不是树
		lset, rset = binSplitData(testData, tree['splitIndex'], tree['splitValue']) #对数据集进行切分
 
		errorNoMerge = np.sum(np.power(lset[:, -1] - tree['leftNode'], 2)) +\
			np.sum(np.power(rset[:, -1] - tree['rightNode'], 2)) #左右两个叶节点的回归误差和, 未进行合并时的误差

		treeMean = (tree['leftNode'] + tree['rightNode']) / 2.0 #两个叶节点的输出均值, 进行合并

		errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2)) #将两个叶节点合并为一个节点时的误差
		if errorMerge < errorNoMerge: #若合成之后的误差比未合成的小,则合并叶节点, 即合成节点的输出值为两个叶节点的均值
			print('merging')
			return treeMean #返回合并之后的叶节点的输出值
		else: #否则不进行合并，返回原来的树
			return tree

	else: #左右节点至少一个子树没有被合并
		return tree


if __name__ == '__main__':
	fileName = r'./ex2.txt'
	data = loadData(fileName)

	tree = createTree(np.array(data), leafType=regLeaf, errType=regErr, ops=(1, 4))
	print(tree, '\n')

	testName = r'./ex2test.txt'
	testdata = loadData(testName)

	pruningTree = postPruning(tree, np.array(testdata))
	print(pruningTree)


