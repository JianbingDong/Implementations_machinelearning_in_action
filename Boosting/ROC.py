"""
绘制ＲＯＣ曲线以及计算ＡＵＣ
"""

import matplotlib.pyplot as plt
import numpy as np


def plotROC(predStrenths, labels):
	"""
	此函数用于绘制ROC(Reciver operating characteristic)曲线
	#arguments:
		predStrenths: np数组或者行向量，代表分类器的预测强度
		labels:list, 真实标签值
	"""
	cur = (1.0, 1.0) #首先将分类阈值设为最小，即将全部样本预测为正例，此时计算出的横、纵坐标值都为１
	ySum = 0.0

	numPosClas = np.sum(np.array(labels) == 1.0) #统计标签中正例的个数,　即 TP + FN
	yStep = 1 / float(numPosClas) #y轴的步进长度，即 1/ (TP + FN)
	xStep = 1 / float(len(labels) - numPosClas)　#x轴的步进长度，即 1/ (TN + FP), TN + FP =　反例个数

	sortedIndicies = predStrenths.argsort() #对数据进行排序，返回排序后的原数据所在的索引

	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(1, 1, 1)

	for index in sortedIndicies.tolist()[0]:　#依次增加阈值，即依次将每个样本预测为反例
		if labels[index] == 1.0: #若该样本实际为正例，则预测结果为FN+1，则此时的真阳率降低
			delX = 0 
			delY = yStep
		else: #若该样本实际为反例，则预测结果为TN+1，则此时的假阳率降低
			delX = xStep
			delY = 0

		ax.plot([cur[0], cur[0] - delX],
				[cur[1], cur[1] - delY],
				c = 'b')
		cur = (cur[0] - delX, cur[0] - delY)

	ax.plot([0, 1], [0, 1], 'b--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve for AdaBoost Horse Colic Detection System')

	ax.axis([0, 1, 0, 1])

	plt.show()
	print("THe AUC (Area Under the Curve is: )", ySum * xStep)
