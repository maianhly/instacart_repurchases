# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, precision_score, recall_score

def classification_metrics(y_test, preds):
    
    df = pd.DataFrame(data=[accuracy_score(y_test, preds), 
                            precision_score(y_test, preds), 
                            recall_score(y_test, preds),
                            roc_auc_score(y_test, preds)],
                     index=['Accuracy', 'Precision', 'Recall', 'AUC'])
    
    return df.T

def plot_confusion_matrix(y_test, preds):
    
    df = pd.DataFrame({'actual': y_test, 'preds': preds})
    cm = pd.crosstab(df['preds'], df['actual'])
    
    sns.heatmap(cm, annot=True, fmt='d')
    plt.ylabel('Predicted label')
    plt.xlabel('True label')

def plot_roc_curve_pres(y_test, preds, figsize=(8, 8), lw=2):
    
    # Store the false positive rate(fpr), true positive rate (tpr) in vectors for use in the graph
    fpr, tpr, _ = roc_curve(y_test, preds)

    # Store the Area Under the Curve (AUC) so we can annotate our graph with theis metric
    roc_auc = auc(fpr, tpr)

    # Plot the ROC Curve
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='#003b55', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color = '#80bfb7', lw = lw, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate', labelpad=10, fontsize=18)
    plt.ylabel('True Positive Rate', labelpad=10, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc = "lower right", fontsize=18)      