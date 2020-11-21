# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:04:35 2020

@author: talha
"""

from simpledatasetloader import SimpleDatsetLoader
from featureExtracting import FeatureExtracting
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from scipy.stats import entropy
from scipy.signal import periodogram
from pywt import wavedec
# from entropy import spectral_entropy
import pyeeg as pyeeg
import csv
import pandas as pd
import math
from scipy import linalg,signal
import matplotlib.pyplot as plt
import scipy.io
from pylab import specgram
from scipy.stats import pearsonr
import numpy as np
from itertools import combinations
# ### Data Loading
lst = np.array(list(combinations(range(32),2)))
print(lst)
dataPaths = "E:\\ECL_VIP\\data"
DEAP_dataset = SimpleDatsetLoader(dataPaths=dataPaths).load_DEAP_BinaryClass()
# print(DEAP_dataset.keys())
# print(np.unique(DEAP_dataset["labels"].ravel(), return_counts=True))
#temp = np.empty(shape=(1,40,2,8064))
#trime_BaseLine = 3 * 128                           # first collection of datapoints
#x = (DEAP_dataset["data"][:, : , 1:2, :])
#print(x.shape)
#x = x.reshape(40,8064)
#x = np.reshape(x, x.size)
#print(x.shape)
#y = (DEAP_dataset["data"][:, : , 2:3, :])
#print(y.shape)
#y = y.reshape(40,8064)
#y = np.reshape(y, y.size)
#print(y.shape)
#cc=np.correlate(x,y,'full')
#print(cc) 
#corr, _ = pearsonr(x, y)
#print('Pearsons correlation: %.3f' % corr)
a = np.array([[0,0]])
for i in range(0,len(lst)):
    x = (DEAP_dataset["data"][:, : , lst[i][0]:lst[i][0]+1, :])
    x = x.reshape(40,8064)
    x = np.reshape(x, x.size)
    y = (DEAP_dataset["data"][:, : , lst[i][1]:lst[i][1]+1, :])
    y = y.reshape(40,8064)
    y = np.reshape(y, y.size)
    corr, _ = pearsonr(x, y)    
    if corr>=0.5:
        b=np.append (a, [[lst[i][0],lst[i][1]]],axis=0)
        a=b
        
#a = np.array([0])
#for i in range(0,32):
#    if i==31:
#        break
#    x = (DEAP_dataset["data"][:, : , 0:1, :])
#    x = x.reshape(40,8064)
#    x = np.reshape(x, x.size)
#    y = (DEAP_dataset["data"][:, : , i+1:i+2, :])
#    y = y.reshape(40,8064)
#    y = np.reshape(y, y.size)
#    corr, _ = pearsonr(x, y)
#    print('Pearsons correlation: %.3f' % corr)
#    if corr>=0:
#        b=np.append (a, [i+1])
#        a=b
#a = np.delete(a, [0]) 
print(a)
 