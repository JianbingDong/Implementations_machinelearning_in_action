"""
此脚本用于使用 疝气病数据集来预测病马的死亡率
@author: Jianbing Dong
"""


from functions import *

def classifyVector(inX, weights):
	"""
	此函数用于对单个输入向量进行分类
	#arguments:
		inX：shape同weights
		weights: shape 为(n, 1)，n指的是特征的个数
	#returns:

	"""
	probability = sigmoid(np.sum(inX * weights)) #计算其前向概率
	if probability > 0.5:
		return 1.0
	else:
		return 0.0

def colicTest():
	"""
	"""
	trainfile = r'./horseColicTraining.txt'
	testfile = r'./horseColicTest.txt'
	train = open(trainfile)
	test = open(testfile)

	errorRate = 0.0
	try:
		#读取训练数据，并创建训练集
		trainSet = []
		trainLabel = []
		for line in train.readlines():
			curLine = line.strip().split('\t')
			lineArr = [] #一个样本的所有特征
			for i in range(len(curLine) - 1):
				lineArr.append(float(curLine[i]))
			trainSet.append(lineArr)
			trainLabel.append(float(curLine[-1]))

		#训练
		weights = gradientAscent_batch(trainSet, trainLabel, batch=None)

		#读取测试数据，并创建测试集
		testSet = []
		testLabel = []
		for line in test.readlines():
			curLine = line.strip().split('\t')
			lineArr = []
			for i in range(len(curLine) - 1):
				lineArr.append(float(curLine[i]))
			testSet.append(lineArr)
			testLabel.append(float(curLine[-1]))

		#测试
		errorCount = 0.0
		for num in range(len(testSet)):
			if int(classifyVector(testSet[num], weights)) != int(testLabel[num]):
				errorCount += 1
		errorRate = float(errorCount) / len(testSet)
		print("The error rate of %s is: %.3f" %(testfile, errorRate))

	except:
		print("num", num)


	finally:
		train.close()
		test.close()

	return errorRate	


def multiTest():
	"""
	此函数用于对测试集进行多次训练测试，对结果取平均值
	"""
	numTests = 10 #测试10次
	errorSUm = 0.0 #总体误差率

	for num in range(numTests):
		errorSUm += colicTest()

	print("After %d times test, the average error rate is %.3f"
			 %(numTests, float(errorSUm) / numTests))



if __name__ == '__main__':
	# colicTest()
	multiTest()
