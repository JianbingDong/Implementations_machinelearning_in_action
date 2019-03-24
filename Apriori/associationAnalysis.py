"""
此脚本使用Apriori算法来发现频繁项集、发现关联规则
"""

import numpy as np

def genData():
	"""
	用于生成简单仿真数据
	"""
	return [[1, 3, 4],
			[2, 3, 5],
			[1, 2, 3, 5],
			[2, 5]]

def createC1(data):
	"""
	创建只含有一个对象的集合,
	即遍历所有交易记录，找到其中出现过的对象的列表，所以要对 item 变为list, 为每种商品构建一个集合
	"""
	C1 = []
	for transactions in data: #遍历数据集，查看每一条交易记录
		for item in transactions: #遍历一条交易记录，查看每个商品
			if [item] not in C1:
				C1.append([item])

	C1.sort()

	return list(map(frozenset, C1))


def scanD(D, ck, minSupport):
	"""
	扫描数据集，以确定满足最小支持度要求的项集
	#arguments:
		D: 数据集, list of sets
		ck:　候选项集列表
		minSupport:　最小支持度
	#returns:
		retList: 
		supportData:
	"""
	ssCnt = {} #用于记录数据集中各个候选集出现的次数
	for tid in D: #遍历数据集，浏览每一条交易记录, set
		for can in ck: #遍历候选集，浏览每个候选集合, frozenset
			if can.issubset(tid): #此条交易记录包含候选集合
				if ssCnt.get(can) is None:#统计候选集出现的次数
					ssCnt[can] = 1
				else:
					ssCnt[can] += 1

	numItems = float(len(D)) #交易的总条数
	retList = [] #结果列表, 满足要求的候选集
	supportData = {} #记录了每个满足要求的候选集的支持度
	for key in ssCnt: #读取每个出现过的候选集
		support = ssCnt[key] / numItems #计算该候选集的支持度
		if support >= minSupport: #若大于最小支持度要求
			retList.insert(0, key) #在列表首部插入该记录
		supportData[key] = support #记录该候选集的支持度

	return retList, supportData


def aprioriGen(Lk, k):
	"""
	创建包含k项的候选集 Ck
	#arguments:
		Lk: 频繁项集列表, 对应每个项有k-1个商品　
		k: integer
	#returns:

	"""
	retList = []
	lenLk = len(Lk)

	for i in range(lenLk):
		for j in range(i+1, lenLk):
			L1 = list(Lk[i])[:k-2] #先将frozenset 转为 list, 用于判断前k-2项是否相同
			L2 = list(Lk[j])[:k-2]
			L1.sort()
			L2.sort()

			if L1 == L2:
				retList.append(Lk[i] | Lk[j])

	return retList


def apriori(data, minSupport):
	"""
	"""
	C1 = createC1(data)
	D = list(map(set, data))

	L1, supportData = scanD(D, C1, minSupport)

	L = [L1]
	k = 2

	while (len(L[k-2]) > 0):
		Ck = aprioriGen(L[k-2], k)
		Lk, supportK = scanD(D, Ck, minSupport)
		supportData.update(supportK)

		L.append(Lk)

		k += 1

	return L, supportData



if __name__ == '__main__':
	data = genData()
	# C1 = createC1(data)
	
	# D = list(map(set, data))
	# L1, supportData0 = scanD(D, C1, 0.5)
	# print(L1)

	L, supportData = apriori(data, 0.5)
	print(L)
	# print(supportData)