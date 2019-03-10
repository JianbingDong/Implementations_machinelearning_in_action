"""
此脚本用于创建模型树，即在CART树的叶节点上，不再是直接输出叶节点的对应数据的标签均值，而是构建一个回归模型
"""

from treeRegression import *

def linearSolver(dataset):
	"""
	用于求解线性回归模型，用最小二乘法的求解公式来求线性模型的参数
	#dataset: np.array with shape [numsamples, featureDim]
	"""
	m, n = dataset.shape

	X = np.mat(np.ones((m, n)))
	Y = np.ones((m, 1))

	X[:, 1:] = dataset[:, : -1] #将数据集扩充一列，其值为１，用作线性方程中的截距ｂ

	Y[:, 0] = dataset[:, -1] #取出数据集的标签值

	xTx = X.transpose() * X

	if np.linalg.det(xTx) == 0:
		raise ValueError("This matrix is singular, cannot do inverse, \n\
						try increasing the second value of ops")
	weights = xTx.I * (X.transpose() * Y)

	return weights, X, Y


def modelLeaf(dataset):
	"""
	用于生成模型树的叶节点, 此时叶节点中存放的是线性模型的回归参数
	"""
	ws, _, _ = linearSolver(dataset)

	return ws

def modelErr(dataset):
	"""
	用于计算模型树的预测误差
	"""
	ws, x, y = linearSolver(dataset)
	yHat = x * ws #计算模型的输出值

	return np.sum(np.power(y - yHat, 2)) #误差的平方和





if __name__ == '__main__':
	fileName = r'./exp2.txt'
	data = loadData(fileName)

	modelTree = createTree(np.array(data), modelLeaf, modelErr, ops=(1, 10))
	print(modelTree)