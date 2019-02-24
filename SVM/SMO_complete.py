"""
此脚本用于实现完整版SMO算法
"""

import numpy as np
from Svm_simple import selectJrand, clipAlpha

class OptStruct(Object):
	"""
	创建一个类型来存储重要的值
	"""
	def __init__(self, dataMat, labelMat, C, epsilon):
		self.dataMat = dataMat #np.mat
		self.labelMat = labelMat #np.mat have been transposed.
		self.C = C #惩罚参数
		self.epsilon = epsilon #校核是否满足KKT条件的epsilon

		self.m = dataMat.shape[0] #样本个数
		self.alphas = np.zeros((self.m, 1)) #初始化m个alpha为０
		self.b = 0　#分隔超平面的偏置
		self.ECache = np.zeros((self.m, 2)) #E_i缓存矩阵，第一列为　E_i缓存是否有效的标志(有效意味着该值已经被计算好)，第二列为实际的eta值


def calcEi(oS, i):
	"""
	计算E_i, 公式7.105
	#arguments:
		oS: OptStruct object
		i: the ith sample
	#returns:
		eta_i
	"""
	gx_i = float(np.sum(np.multiply(np.multiply(oS.alphas, oS.labelMat),
		np.matmul(oS.dataMat, oS.dataMat[i].transpose())))) + oS.b #《统计学习方法》公式7.104
	E_i = gx_i - float(oS.labelMat[i]) #公式7.105	
	return E_i


def selectJ(oS, i, E_i):
	"""
	用于选择第二个alpha,　即alpha_j,　通过｜E1 - E2|最大的方式来选择alpha_j
	#arguments:
		oS: OptStruct Object,
		i: 选择的alpha_i的下标
		E_i: alpha_i的E_i
	"""
	maxJ = -1 #最大的alpha_j
	maxDeltaE = 0 #最大的差值
	E_j = 0 #最大差值对应的Ｅ_j

	oS.ECache[i] = [1, E_i] #Ｅ_i已被计算出来，将其存入缓存矩阵中，并设置对应的标志位有效
	validECachelist = np.nonzero(oS.ECache[:, 0])[0] #返回非零元素的索引值??
	if len(validECachelist) > 1:
		for k in validECachelist:
			if k == i:
				continue
			E_k = calcEi(oS, k) #计算E_k
			deltaE = np.abs(E_i - E_k) #计算|E_i - E_k|
			if deltaE > maxDeltaE:
				maxJ = k
				maxDeltaE = deltaE
				E_j = E_k

		return maxJ, E_j

	else:
		j = selectJrand(i, oS.m)
		E_j = calcEi(oS, j)

		return j, E_j

def updateEj(oS, j):
	"""
	用于计算误差值，并存储在缓存矩阵当中
	"""
	Ej = calcEi(oS, j)
	oS.ECache[j] = [1, Ej]


def innerL(i, oS):
	"""
	"""
	Ei = calcEi(oS, i)
	if ((os.labelMat[i] * Ei < - oS.epsilon) and (oS.alphas[i] < oS.C))\
	 or ((os.labelMat[i] * Ei > oS.epsilon) and (oS.alphas[i] > 0)):
	 	j, Ej = selectJ(oS, i, Ei)
	 	alphaI_old = oS.alphas[i].copy()
	 	alphaJ_old = oS.alphas[j].copy()

	 	if (oS.labelMat[i] != oS.labelMat[j]):
	 		L = max(0, oS.alphas[j] - os.alphas[i])
	 		H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
	 	else:
	 		L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
	 		H = min(oS.C, oS.alphas[j] + oS.alphas[i])

	 	if L == H:
	 		print("L == H")
	 		return 0

	 	eta = 2.0 * oS.dataMat[i, :] * oS.dataMat[j, :].transpose() -\
	 		 oS.dataMat[i, :] * oS.dataMat[i, :].transpose() -\
	 		 oS.dataMat[j, :] * oS.dataMat[j, :].transpose()

	 	if eta >= 0:
	 		print('eta >= 0')
	 		return 0

	 	os.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
	 	os.alphas[j] = clipAlpha(oS.alphas[j], H, L)

	 	updateEj(oS, j)

	 	if (np.abs(oS.alphas[j] - alphaJ_old) < 1e-5):
	 		print("J not moving enough")
	 		return 0
	 	oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] *\
	 		(alphaJ_old - oS.alphas[j])

	 	updateEj(oS, i)

	 	b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaI_old) *\
	 		oS.dataMat[i, :] * oS.dataMat[i, :].transpose() - oS.labelMat[j] *\
	 		(oS.alphas[j] - alphaJ_old) * oS.labelMat[j, :] * oS.dataMat[j, :].transpose()
	 	b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaI_old) *\
	 		oS.dataMat[i, :] * oS.dataMat[j, :].transpose() - oS.labelMat[j] *\
	 		(oS.alphas[j] - alphaJ_old) * oS.labelMat[j, :] * oS.dataMat[j, :].transpose()
	 	if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
	 		oS.b = b1
	 	elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
	 		oS.b = b2
	 	else:
	 		oS.b = (b1 + b2) / 2

	 	return 1

	else:
		return 0

		
