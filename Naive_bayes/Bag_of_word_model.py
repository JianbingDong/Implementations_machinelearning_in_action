"""
This script is used for realization of foundamental functions.
@author Jianbing Dong
"""

"""
此脚本中将词汇出现的次数作为特征，为 词袋模型（bag-of-words model)
"""

import numpy as np


def loadDataset():
	"""
	This function is used to create dataset.
	"""
	#人为创造的数据集，此数据集中包含6条文档，可以当做6个文本
	postingList = [['my', 'dog', 'has', 'flea', \
					'problems', 'help', 'please'],
					['maybe', 'not', 'take', 'him',\
					'to', 'dog', 'park', 'stupid'],
					['my', 'dalmation', 'is', 'so',\
					'cute', 'I', 'love', 'him'],
					['stop', 'posting', 'stupid', 'worthless', 'garbage'],
					['mr', 'licks', 'ate', 'my', 'steak', 'how', \
					'to', 'stop', 'him'],
					['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]


	classVec = [0, 1, 0, 1, 0, 1] #标签，1代表侮辱性文字，0代表正常

	return postingList, classVec


def createVocabList(dataset):
	"""
	This function is used to create vocabulary based the dataset.
	#arguments:
		dataset: list.
	#returns:
		vocabList: list, all different word in a list.
	vocablist在一次创建之后，其中词的顺序就是固定的。多次创建，可能会不同
	"""
	#way-1
	vocabList = []
	for document in dataset:
		vocabList.extend(document)

	vocabList = set(vocabList) #用set去重

	#way-2
	# vocabList = set([])
	# for document in dataset:
	# 	vocabList = vocabList | set(document)

	return list(vocabList)

def bagofWords2Vec(vocablist, inputset):
	"""
	#arguments：
		vocablist：词汇表，之前创建的含所有词汇的list
		inputset：某个文档
	#returns：
		workVec：文档向量，若文档中的词出现在词汇表中，则为1，否则为0
		用词出现的次数作为特征
	"""
	wordVec = [0] * len(vocablist) #创建一个与词汇表等长的0向量
	for word in inputset: #此过程用于判断，词汇表中的单词在文档中是否出现，出现则记为1
		if word in vocablist:
			wordVec[vocablist.index(word)] += 1
		else:
			print("the word %s is not in my vocabulary!" %word)

	return wordVec 


def trainNaiveBayes(trainMatrix, trainCategory):
	"""
	#arguments:
		trainMatrix: 文档矩阵，[[0, 1, ..], [..]]
			为一个list，其中每个元素也是一个list，指的是每篇文档的wordVec， 
			wordVec的长度与词汇表长度相同
		trainCategory: 文档类别，形式类似于loadDataset函数的第二个返回值
	"""
	eposilon = 1e-16
	numTrainDocs = len(trainMatrix) #获取训练文档的数量
	numWords = len(trainMatrix[0]) #获取wordVec的长度

	pAbusive = np.sum(trainCategory) / float(numTrainDocs) #计算类别为1的概率，即 p(c = 1)

	p0num = np.zeros(numWords) #构建与wordVec相同长度的零向量
	p1num = np.zeros(numWords) #同上
	p0Denom = 0.0
	p1Denom = 0.0

	for i in range(numTrainDocs): #对每篇训练文档
		if trainCategory[i] == 1: #判断该文档的类别是否为1
			p1num += trainMatrix[i] #将该文档中词出现的频次加到 p1num 上，即f(w1 | c = 1), f(w2 | c=1), ...
			# p1Denom += np.sum(trainMatrix[i]) #统计该文档总共出现了多少个词，为书上源代码
		else:
			p0num += trainMatrix[i] #将该文档中词出现的频次加到 p0num 上，即f(w1 | c = 0), f(w2 | c=0), ...
			# p0Denom += np.sum(trainMatrix[i]) #统计该文档中总共出现了多少个词，为书上源代码

	p0Denom += np.sum(p0num) #统计0类文档中总共出现了多少个词，即f(w | c = 0)
	p1Denom += np.sum(p1num) #统计1类文档中总共出现了多少个词，即f(w | c = 1)

	p1Vect = p1num / p1Denom #计算1类文档中每个词的概率，即p(w1 | c = 1), p(w2 | c=1), ...
	p0Vect = p0num / p0Denom #计算0类文档中每个词的概率，即p(w1 | c = 0), p(w2 | c=0), ...

	p1Vect = np.log(p1Vect + eposilon) #为避免数值下溢，将概率转换为对数，注意之后的概率相乘应该转变为概率相加
	p0Vect = np.log(p0Vect + eposilon) 

	return p0Vect, p1Vect, pAbusive


def classifyNaiveBayes(vec2classify, p0vec, p1vec, pclass1):
	"""
	#arguments:
		vec2classify: 根据词汇表转换出的文档向量
		p0vec: 类别为0的对数概率向量，长度与vec2classify相同
		p1vec: 类别为1的对数概率向量，长度同上
		pclass1: integer, p(c = 1)
	#returns:

	"""
	p1 = np.sum(vec2classify * p1vec) + np.log(pclass1) #依据条件独立性假设，从概率向量中取值出来，利用了文档向量的0-n特性
	p0 = np.sum(vec2classify * p0vec) + np.log(1.0 - pclass1)

	if p1 > p0:
		return 1
	else:
		return 0


def testing_NaiveBayes():
	"""
	此函数用于统筹使用之前定义的函数，相当于主函数
	"""
	postinglist, labels = loadDataset() #载入文档数据及其标签
	myvocablist = createVocabList(postinglist) #依据文档数据构建词汇表
	trainmat = []
	for postingDoc in postinglist:
		trainmat.append(bagofWords2Vec(myvocablist, postingDoc)) #将文档依据词汇表转换为文档矩阵
	p0v, p1v, pAb = trainNaiveBayes(trainmat, labels) #计算条件概率，已取对数

	testentry = ['love', 'my', 'dalmation'] #测试文档
	thisDoc = bagofWords2Vec(myvocablist, testentry) #依据词汇表将文档转为文档向量
	result = classifyNaiveBayes(thisDoc, p0v, p1v, pAb) #对该文档向量进行分类
	print(testentry, "classified as %d\n" %result) #显示结果

	testentry1 = ['stupid', 'garbage'] 
	thisDoc1 = bagofWords2Vec(myvocablist, testentry1) 
	resutl1 = classifyNaiveBayes(thisDoc1, p0v, p1v, pAb)
	print(testentry1, 'classified as %d\n' %resutl1)




if __name__ == '__main__':
	testing_NaiveBayes()