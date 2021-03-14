# -*- coding: utf-8 -*-
import numpy as np
import logging
import os

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc

PATH = 'CURVE'

def LoadData(path):
    with open(path,'r+') as f:
        Feature = np.loadtxt(f,delimiter=',',skiprows=0)
        return np.array(Feature)

plus = lambda x:x+1e-6 if x==0 else x

def Eval(pre_logits,label):
    pre_logits = np.squeeze(pre_logits)
    TP = pre_logits[pre_logits == label]
    TP = TP[TP == 0]
    TP = len(TP)
    
    FP = pre_logits[pre_logits != label]
    FP = FP[FP == 1]
    FP = len(FP)
    
    TN = pre_logits[pre_logits == label]
    TN = TN[TN == 1]
    TN = len(TN)
    
    FN = pre_logits[pre_logits != label]
    FN = FN[FN == 0]
    FN = len(FN)
    Acc = (TP+TN)/plus(TP+FP+TN+FN)
    Se = TP/plus(TP+FN)
    Sp = TN/plus(TN+FP)
    deno = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    Mcc = (TP*TN-FP*FN)/plus(deno)**0.5
    return Acc,Se,Sp,Mcc

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        #log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        log_file = os.path.join(root, phase + '.log')
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def AucPlt1(name,label,Score):
    fpr1,tpr1,threshold1 = roc_curve(label,Score)
    roc_auc1 = auc(fpr1,tpr1)
    plt.figure()
    plt.figure(figsize=(7,6))
    plt.plot(fpr1,tpr1,color = 'orange',linewidth=1,label='Mismatch ROC curve(AUC= %0.4f)'%roc_auc1)
    plt.plot([0,1],[0,1],color = 'black',linewidth=1,linestyle='-.')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.legend(loc="lower right")
    plt.savefig('./'+PATH+'/'+name.replace(' ', '_') + '.jpg')


def Int2Roman(num):
    # 确定个十百千各自位置上的0~9对应罗马字母
    c = {
        'g': ('', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX'),
        's': ('', 'X', 'XX', 'XXX', 'XL', 'L', 'LX', 'LXX', 'LXXX', 'XC'),
        'b': ('', 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM'),
        'q': ('', 'M', 'MM', 'MMM')
    }
    roman = []
    # 用整除和取余获得个十百千对应的数字
    roman.append(c['q'][num // 1000])
    roman.append(c['b'][(num // 100) % 10])
    roman.append(c['s'][(num // 10) % 10])
    roman.append(c['g'][num % 10])
    return ''.join(roman)


def AucPlt2(name,train_label,train_Score,test_label,test_Score):
    fpr1,tpr1,threshold1 = roc_curve(train_label,train_Score)
    roc_auc1 = auc(fpr1,tpr1)
    fpr2,tpr2,threshold2 = roc_curve(test_label,test_Score)
    roc_auc2 = auc(fpr2,tpr2)
    plt.figure()
    plt.figure(figsize=(7,6))
    plt.plot(fpr1,tpr1,color = 'orange',linewidth=1,label='Layer I ROC curve(AUC= %0.4f)'%roc_auc1)
    plt.plot(fpr2,tpr2,color = 'green',linewidth=1,label='Layer II ROC curve (AUC= %0.4f)'%roc_auc2)
    plt.plot([0,1],[0,1],color = 'black',linewidth=1,linestyle='-.')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.legend(loc="lower right")
    plt.savefig('./'+PATH+'/'+name.replace(' ', '_') + '.jpg')











