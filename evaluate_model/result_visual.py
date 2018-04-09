#!/usr/bin/env python
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import cPickle
from scipy import interp
import numpy as np
import itertools
from itertools import cycle

file_name = 'for_roc_pyrimidines_classification.pickle'

def label_one_hot(label_array):
	from sklearn.preprocessing import OneHotEncoder
	enc=OneHotEncoder()
	label_list=[]
	for i in range(len(label_array)):
		label_list.append([label_array[i]])
	return enc.fit_transform(label_list).toarray()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm_ini = cm
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(dpi = 350, figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm_ini.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm_ini[i, j],
                 horizontalalignment="center",
                 color="white" if cm_ini[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_PRC(y_score, y_test, classes, title='Precision_Recall_Curve'):
    lw = 2
    precision = dict()
    recall = dict()
    average_precision = dict()
    if len(np.shape(y_test)) == 1:
        y_test = np.array(label_one_hot(y_test))
    y_score = np.array(y_score)
    for i in range(len(y_test[0])):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
            y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,
        average="micro")

    # # Plot Precision-Recall curve
    # plt.clf()
    # plt.plot(recall[0], precision[0], lw=lw, color='navy',
    #          label='Precision-Recall curve')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    # plt.legend(loc="lower left")
    # # plt.show()
    # plt.savefig('PRC.jpg')

    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    # Plot Precision-Recall curve for each class
    plt.figure(dpi = 350, figsize=(8,6))
    plt.clf()
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    for i, color in zip(range(len(y_test[0])), colors):
        plt.plot(recall[i], precision[i], color=color, lw=lw,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(classes[i], average_precision[i]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(title+'.jpg',bbox_inches = 'tight')


def plot_ROC(y_score, y_test, classes, title='ROC'):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if len(np.shape(y_test)) == 1:
        y_test = np.array(label_one_hot(y_test))
    y_score = np.array(y_score)
    for i in range(len(y_test[0])):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(y_test[0]))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(y_test[0])):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(y_test[0])
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(dpi = 350, figsize=(8,6))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    for i, color in zip(range(len(y_test[0])), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(title+'.jpg',bbox_inches = 'tight')


def draw_cm():
    cm_dl_purine_usual_thickness = np.array([[1363, 451], 
        [394, 2013]])
    cm_dl_pyrimidine_usual_thickness = np.array([[1639, 424], 
        [382, 1678]])
    cm_dl_7_classes_usual_thickness = np.array([[5757, 0, 0, 0, 0, 0, 5], 
        [2, 1245, 33, 38, 526, 0, 0], 
        [2, 9, 1506, 477, 11, 0, 3],
        [1, 10, 408, 1663, 20, 0, 6], 
        [2, 301, 64, 39, 1950, 0, 1], 
        [2, 1, 1, 0, 2, 8202, 3],
        [9, 1, 1, 0, 0, 2, 8334]])
    cm_dl_purine_vs_pyrimidine_usual_thickness = np.array([[4131, 156],
        [87, 3969]])
    cm_dl_5_classes_usual_thickness = np.array([[5753, 0, 0, 2, 7],
        [1, 4064, 135, 0, 1],
        [2, 71, 4035, 0, 8], 
        [4, 2, 1, 8200, 4], 
        [1, 2, 1, 2, 8341]])
    plt.figure()
    plot_confusion_matrix(cm_dl_purine_usual_thickness, ['A', 'G'], 
        normalize=True, title='cm_dl_purine_usual_thickness')
    plt.savefig('cm_dl_purine_usual_thickness.jpg', bbox_inches = 'tight')
    plt.figure()
    plot_confusion_matrix(cm_dl_pyrimidine_usual_thickness, ['C', 'U'], 
        normalize=True, title='cm_dl_pyrimidine_usual_thickness')
    plt.savefig('cm_dl_pyrimidine_usual_thickness.jpg', bbox_inches = 'tight')
    plt.figure()
    plot_confusion_matrix(cm_dl_7_classes_usual_thickness, ['Non_site', 'A',
        'U', 'C','G', 'P', 'Ribose'], 
        normalize=True, title='cm_dl_7_classes_usual_thickness')
    plt.savefig('cm_dl_7_classes_usual_thickness.jpg', bbox_inches = 'tight')
    plt.figure()
    plot_confusion_matrix(cm_dl_purine_vs_pyrimidine_usual_thickness, ['Purine', 
        'Pyrimidine'], 
        normalize=True, title='cm_dl_purine_vs_pyrimidine_usual_thickness')
    plt.savefig('cm_dl_purine_vs_pyrimidine_usual_thickness.jpg', bbox_inches = 'tight')
    plt.figure()
    plot_confusion_matrix(cm_dl_5_classes_usual_thickness, ['Non_site', 'Purine',
        'Pyrimidine','P', 'Ribose'], 
        normalize=True, title='cm_dl_5_classes_usual_thickness')
    plt.savefig('cm_dl_5_classes_usual_thickness.jpg', bbox_inches = 'tight')

    # plt.show()

def open_file(file_name):
    with open(file_name, 'r') as f:
        prob, label = cPickle.load(f)
    return prob, label

if __name__ == '__main__':
    # draw_cm()
    prob, label = open_file(file_name)
    # plot_ROC(prob, label, ['Non_site', 'Purine', 'Pyrimidine','P', 'Ribose'], 'ROC_5_classes')
    # plot_PRC(prob, label, ['Non_site', 'Purine', 'Pyrimidine','P', 'Ribose'], 'PRC_5_classes')
    # plot_ROC(prob, label, ['Non_site', 'A',
    #     'U', 'C','G', 'P', 'Ribose'], 'ROC_7_classes')
    # plot_PRC(prob, label, ['Non_site', 'A',
    #     'U', 'C','G', 'P', 'Ribose'], 'PRC_7_classes')
    # plot_ROC(prob, label, ['A', 'G'], 'ROC_purine')
    # plot_PRC(prob, label, ['A', 'G'], 'PRC_purine')
    plot_ROC(prob, label, ['U', 'C'], 'ROC_pyrimidine')
    plot_PRC(prob, label, ['U', 'C'], 'PRC_pyrimidine')
