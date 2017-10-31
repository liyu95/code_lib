#!/usr/bin/env python

import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle

'''
The input of the class should be a dataframe with each row representing
a data point and each column representing a feature. The last column should
be the 'label'. 
'''

class SVMEvaluator(object):
	"""check the performance of SVM"""
	def __init__(self, dataframe):
		super(SVMEvaluator, self).__init__()
		self.label = dataframe['label'].values
		self.features = dataframe.iloc[:, 1:-1].values
		self.n_classes = len(set(list(self.label)))
		self.label_one_hot = label_binarize(self.label, range(self.n_classes))

	def cross_validation(self, kernel='rbf', C=1, cv=5):
		clf = svm.SVC(kernel=kernel, C=C)
		scores = cross_val_score(clf, self.features, self.label, cv=cv)
		print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	def run_classifier(self, seed=6, kernel='rbf', C=1):
		
		X_train, X_test, y_train, y_test = train_test_split(self.features, 
			self.label_one_hot, test_size=0.9, random_state=seed)
		classifier = OneVsRestClassifier(svm.SVC(kernel=kernel, 
			C=1, probability=True, random_state=seed))
		y_score = classifier.fit(X_train, y_train).decision_function(X_test)
		return y_score, y_test

	def plot_ROC(self, y_score, y_test):
		# Compute ROC curve and ROC area for each class
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for i in range(self.n_classes):
		    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
		    roc_auc[i] = auc(fpr[i], tpr[i])

		# Compute micro-average ROC curve and ROC area
		fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
		roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

		lw = 2

		# Compute macro-average ROC curve and ROC area

		# First aggregate all false positive rates
		all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))

		# Then interpolate all ROC curves at this points
		mean_tpr = np.zeros_like(all_fpr)
		for i in range(self.n_classes):
		    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

		# Finally average it and compute AUC
		mean_tpr /= self.n_classes

		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr
		roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

		# Plot all ROC curves
		plt.figure()
		plt.plot(fpr["micro"], tpr["micro"],
		         label='micro-average ROC curve (area = {0:0.2f})'
		               ''.format(roc_auc["micro"]),
		         color='deeppink', linestyle=':', linewidth=4)

		plt.plot(fpr["macro"], tpr["macro"],
		         label='macro-average ROC curve (area = {0:0.2f})'
		               ''.format(roc_auc["macro"]),
		         color='navy', linestyle=':', linewidth=4)

		colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
		for i, color in zip(range(self.n_classes), colors):
		    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
		             label='ROC curve of class {0} (area = {1:0.2f})'
		             ''.format(i, roc_auc[i]))

		plt.plot([0, 1], [0, 1], 'k--', lw=lw)
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Some extension of Receiver operating characteristic to multi-class')
		plt.legend(loc="lower right")
		plt.show()

	def plot_PRC(self, y_score, y_test):
		lw = 2
		precision = dict()
		recall = dict()
		average_precision = dict()
		for i in range(self.n_classes):
			precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
				y_score[:, i])
			average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

		precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
			y_score.ravel())
		average_precision["micro"] = average_precision_score(y_test, y_score,
			average="micro")

		# Plot Precision-Recall curve
		plt.clf()
		plt.plot(recall[0], precision[0], lw=lw, color='navy',
		         label='Precision-Recall curve')
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])
		plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
		plt.legend(loc="lower left")
		plt.show()

		colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
		# Plot Precision-Recall curve for each class
		plt.clf()
		plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
		         label='micro-average Precision-recall curve (area = {0:0.2f})'
		               ''.format(average_precision["micro"]))
		for i, color in zip(range(self.n_classes), colors):
		    plt.plot(recall[i], precision[i], color=color, lw=lw,
		             label='Precision-recall curve of class {0} (area = {1:0.2f})'
		                   ''.format(i, average_precision[i]))
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.title('Extension of Precision-Recall curve to multi-class')
		plt.legend(loc="lower right")
		plt.show()

	def Polt_ROC_PRC(self, seed=6, kernel='rbf', C=1):
		y_score, y_test = self.run_classifier(seed, kernel, C)
		self.plot_ROC(y_score, y_test)
		self.plot_PRC(y_score, y_test)

if __name__ == '__main__':
	print "We are in the main function."