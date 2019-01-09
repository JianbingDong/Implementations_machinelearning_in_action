"""
此脚本用于对从个人广告中获取区域倾向
"""
import feedparser
import operator
from Set_of_word_model import *
from Bag_of_word_model import bagofWords2Vec
import re
import random

def calcMostFreq(vocabList, fullText):
	"""
	用于计算出现频率最多的词
	#arguments:
		vocablist: 词汇列表，list，里面没有重复的词
		fullText: 包含所有词的list，有的词有重复
	#returns:
		返回出现频率最多的前30个词
	"""
	freqDict = {}
	for token in vocabList:
		freqDict[token] = fullText.count(token) #统计token在fulltext中出现的次数

	#对dict进行排序，按照value的大小逆序排列，返回dict
	sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)

	return sortedFreq[:30] #返回出现频率最多的30个词 组织形式为{‘word’, frequency, ...}


def textParse(bigstring):
	"""
	接受一个大字符串，并将其解析为一个字符串列表
	此时解析条件是 不含数字，长度大于2，全为小写
	"""
	listofTokens = re.split(r'\W*', bigstring)
	return [tok.lower() for tok in listofTokens if (len(tok) > 2)]


def localWords(feed1, feed0):
	"""
	"""
	doclist = [] #用于存储所有文档
	classlist = [] #用于存储文档标签
	fullText = [] #用于存储所有单词，单词有重复
	minLen = min(len(feed1['entries']), len(feed0['entries'])) #rss源最小条数

	for i in range(minLen):
		wordlist  = textParse(feed1['entries'][i]['summary']) #从str中解析出字符串列表
		doclist.append(wordlist) #将list添加为doclist的一个元素
		fullText.extend(wordlist) #将list与fulltext相拼接
		classlist.append(1) #为1类添加一个计数

		wordlist = textParse(feed0['entries'][i]['summary']) #从str中解析出字符串列表
		doclist.append(wordlist) #将list添加为doclist的一个元素
		fullText.extend(wordlist) #将list与fulltext相拼接
		classlist.append(0) #为0类添加一个计数

	vocabList = createVocabList(doclist) #从doclist（所有文档）创建词汇表，返回值为不重复的所有词
	top30Words = calcMostFreq(vocabList, fullText) #根据词汇表统计全文中所有词出现的次数，并返回频率最高的前30词

	#从词汇表中删除出现频率最高的词
	for pairW in top30Words:
		if pairW[0] in vocabList: #若该词出现在词汇表中
			vocabList.remove(pairW[0]) #从词汇表中删掉该词

	#构建训练集和测试集，共2× minLen条数据，因为上面的for训练里面分别解析feed0和feed1各minLen次
	trainingSet = list(range(2 * minLen)) # [0, 1, ..., 2* minLen - 1] 的列表，长度为2*minLen
	testSet = []

	#随机选取20条rss源作为测试集，其余的作为训练集
	for i in range(5): #因找到的rss最多只有10条数据，因此，选用5条作为测试集
		randIndex = int(random.uniform(0, len(trainingSet))) #按正态分布从[0, len(trainingSet))中取一个值，并取整
		testSet.append(trainingSet[randIndex]) #将该随机数代表的文档标号添加到testSet中
		del(trainingSet[randIndex]) #从trainingSet中删除该文档标号

	#构建训练矩阵
	trainMat = []
	trainClasses = []
	#遍历训练集中的所有文档
	for docIndex in trainingSet:
		trainMat.append(bagofWords2Vec(vocabList, doclist[docIndex])) #从每篇文档中依据词汇表创建词袋向量
		trainClasses.append(classlist[docIndex]) #添加标签

	p0v, p1v, pSpam = trainNaiveBayes(trainMat, trainClasses) #使用训练矩阵和训练标签计算 p(wi | c=0)向量 及 p(wi | c=1)向量 及 p(c=1)概率
	
	errorcount = 0
	for docIndex in testSet:
		wordVector = bagofWords2Vec(vocabList, doclist[docIndex]) #依据词汇表，从测试集中的每篇文档返回词袋向量
		if classifyNaiveBayes(wordVector, p0v, p1v, pSpam) != classlist[docIndex]: #判断是否分类错误
			errorcount += 1 #分类错误数加1

	print('The error rate is %.2f' %(float(errorcount) / len(testSet))) #计算错误率

	return vocabList, p0v, p1v



if __name__ == '__main__':
	index1 = r'http://blog.sina.com.cn/rss/1092672395.xml'
	index2 = r'http://blog.sina.com.cn/rss/1258463770.xml'

	feed1 = feedparser.parse(index1)
	feed2 = feedparser.parse(index2)

	vocablist, p0v, p1v = localWords(feed1, feed2)

	# minLen = min(len(feed1['entries']), len(feed2['entries']))
	# print(len(feed1['entries']))
	# print(len(feed2['entries']))