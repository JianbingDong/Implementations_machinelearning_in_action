"""
使用ＥＭ算法来求解高斯混合模型参数
代码实现
"""

import numpy as np

PI = 3.141592653

def generateData(numK, N):
	"""
	生成高斯混合模型产生的数据，此过程会保存高斯混合模型的各个参数，便于后期与ＥＭ算法比较
	#arguments:
		numK: integer, 制定有多少个高斯模型混合而成
		N: integer, 一共采样Ｎ个样本
	#returns:
		Y: 1-dim list, 采样生成的Ｎ条数据
	"""
	nums = np.random.randint(low=1, high=numK*2, size=numK) 
	alphaK = nums / sum(nums) #产生选择每个高斯模型的随机概率 alphaK

	mius = np.random.randint(low=0, high=numK * 3, size=numK) #各个高斯模型的均值
	sigmas = np.abs(np.random.randn(numK)) #各个高斯模型的方差
	
	Y = [] #用于记录采样结果
	for i in range(N): #进行N次采样
		index = int(np.random.choice(a=range(numK), size=1, p=alphaK)) #依概率alphaK 随机选取一个高斯分布
		miu, sigma = list(zip(mius, sigmas))[index] 
		y = miu + np.random.randn() * sigma #从 N(miu, sigma^2)中随机采样一个数

		Y.append(y)

	with open(r'./generateData.txt', 'w') as file:
		file.write("generateData statistics\n\n")

		file.write("alpha: ")
		for alpha in alphaK:
			file.write(str(alpha) + '\t')
		file.write('\n')

		file.write('miu: ')
		for miu in mius:
			file.write(str(miu) + '\t')
		file.write('\n')

		file.write('sigma: ')
		for sigma in sigmas:
			file.write(str(sigma) + '\t')
		file.write('\n')


	return Y



def initParameters(K):
	"""
	此函数用于初始化高斯混合模型的参数
	#arguments:
		K: integer, 高斯模型的个数
	#returns:
		alphaK, mius, sigmas: 初始化后的各个参数,　皆为1-dim list
	"""
	alphaK = [1/K] * K #等概率初始化每个alpha
	# mius = [1] * K #全1 初始化均值
	sigmas = [1] * K #全１初始化标准差

	mius = list(np.random.randn(K))

	return alphaK, mius, sigmas


def EM_algorithm(Y, K, ep1):
	"""
	ＥＭ算法求解
	#arguments:
		Y: 1-dim list, 观测数据y1, y2, y3....yn
		K: integer, 总共有Ｋ个高斯模型
		ep1, ep2: float, 用于判断ＥＭ算法是否收敛
	"""
	alphaK_i, mius_i, sigmas_i = initParameters(K) #初始化参数

	#转换为矩阵类型
	Y = np.mat(Y) #[1, N]
	alphaK_i = np.mat(alphaK_i) #[1, K]
	mius_i = np.mat(mius_i) #[1, K]
	sigmas_i = np.mat(sigmas_i) #[1, K]
	
	print('miu_init: ', mius_i)
	print("alpha_init: ", alphaK_i)
	print('sigma_init: ', sigmas_i)

	step = 0
	# while step < int(1e3):
	while 1:
		print("\n")
		#E 步，求期望
		gamma = gamma_jk(Y, alphaK_i, mius_i, sigmas_i) #计算给定模型参数下的响应度 

		#Ｍ　步，更新参数，求最大
		miu_new = (Y * gamma) / np.sum(gamma, axis=0)#求出新的均值 [1, K]
		
		alpha_new = np.sum(gamma, axis=0) / Y.shape[-1] #新的alphaK, [1, K]
		

		#求出新的sigma
		Y_miu = np.ones((Y.shape[-1], K)) #[N, K] 矩阵，用于存储 (yj-miuk) ^ 2
		for k in range(K):
			Y_miu[:, k] = np.reshape(np.square(Y.T - mius_i[0, k]), newshape=(Y.shape[-1], ))

		sigma_new = np.sqrt(np.sum(np.array(gamma) * Y_miu, axis=0) / np.sum(gamma, axis=0))

		print('miu_new: ', miu_new)
		print('alpha_new: ', alpha_new)
		print('sigma_new: ', sigma_new)

		#判断是否收敛
		if updataDegree(alpha_new, miu_new, sigma_new, alphaK_i, mius_i, sigmas_i) < ep1:
			return alpha_new, miu_new, sigma_new

		else:
			alphaK_i, mius_i, sigmas_i = alpha_new, miu_new, sigma_new
			step += 1

	return alpha_new, miu_new, sigma_new


def gamma_jk(Y, alphaK, mius, sigmas):
	"""
	根据给定模型参数，计算分模型K对观测数据yj的响应度
	#arguments:
		Y: np.mat [1, N], 观测数据
		alphaK, mius, sigmas: np.mat, [1, K] 当前给定的模型参数
	#returns:
		gamma: np.mat (N, K) 矩阵，记录了分模型ｋ对观测数据ｙｊ的响应度
	"""
	gamma = np.zeros((Y.shape[-1], alphaK.shape[-1])) #生成 (N, K)维度的全０矩阵
	for j in range(Y.shape[-1]): #Ｎ
		for k in range(alphaK.shape[-1]): #K
			gamma[j][k] = alphaK[0, k] * GaussianProbability(Y[0, j], mius[0, k], sigmas[0, k]) #先计算分子

		gamma[j] /= np.sum(gamma[j]) #再除以分子之和，进行归一化

	return np.mat(gamma)


def GaussianProbability(yj, miu, sigma):
	"""
	用于计算给定参数下的高斯分布概率密度
	#arguments:
		yj: 第ｊ个样本
		miu, sigma: 当前高斯模型的参数
	"""
	phi = (1.0 / np.sqrt(2*PI*np.square(sigma + 1e-8))) * np.exp(-0.5 * np.square((yj - miu) / (sigma + 1e-8)))
	return phi



def updataDegree(alpahK_new, miu_new, sigma_new, alpahK_old, miu_old, sigma_old):
	"""
	此函数用于计算参数更新程度
	#arguments:
		alpahK, miu, sigma, 每个都是[1, K]维度的np.mat
		alphaK, miu, sigma, 每个都是[1, K]维度的np.mat
	#return:
		degree: float, 本次参数的更新程度 
	"""
	degree = 0.0
	degree += np.sum(np.square(alpahK_new - alpahK_old))
	degree += np.sum(np.square(miu_new - miu_old))
	degree += np.sum(np.square(sigma_new - sigma_old))

	return degree


if __name__ == '__main__':
	K, N = 6, 10
	Y = generateData(K, N)

	alphas, mius, sigmas = EM_algorithm(Y, K, ep1=1e-8)

	with open(r'./generateData.txt', 'a') as file:
		file.write("\n EM algorithm statistics\n")

		file.write("alpha: ")
		for alpha in alphas:
			file.write(str(alpha) + '\t')
		file.write('\n')

		file.write('miu: ')
		for miu in mius:
			file.write(str(miu) + '\t')
		file.write('\n')

		file.write('sigma: ')
		for sigma in sigmas:
			file.write(str(sigma) + '\t')
		file.write('\n')

