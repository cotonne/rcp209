#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import genfromtxt

import csv
import collections
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2 as chi2_s
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from matplotlib import cm as colormap
import pydotplus
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


sns.set(style="whitegrid", color_codes=True)


# Reading CSV
filename = 'german.data.csv'
delimiter = ' '
data = []

continuous_values = pd.read_csv(filename, delimiter=delimiter, 
    names=['Installement rate', 'Present residence since', 'Number of existing credits', 'Nb of liable people'],
    usecols=[7, 10, 15, 17],
    dtype='float')
continuous_values = pd.read_csv(filename, delimiter=delimiter, 
    names=['Duration in month', 'Credit amount', 'Installement rate', 'Present residence since', 'Age', 'Number of existing credits', 'Nb of liable people'],
    usecols=[1,4, 7, 10, 12, 15, 17],
    dtype='float')


discrete_values = pd.read_csv(filename,delimiter=delimiter,
    names=['Purpose', 'Personal status and sex', 'Guarantors',
         'Housing', 'Job', 'Telephone', 'Foreigner', 'Credit'],
    usecols=[3,8,9,14,16,18,19,20],
    dtype='S4')
discrete_values = pd.read_csv(filename,delimiter=delimiter,
    names=['Status of checking account', 'Credit history', 
         'Purpose', 'Savings account', 'Present employment since', 
         'Personal status and sex', 'Guarantors', 'Property',
         'Other installment plans', 'Housing', 
         'Job', 'Telephone', 'Foreigner', 'Credit'],
    usecols=[0,2,3,5,6,8,9,11,13,14,16,18,19,20],
    dtype='S4')

credit_values = discrete_values['Credit']
discrete_values = discrete_values.drop(labels='Credit', axis=1)

# Reprojection
data = pd.concat([discrete_values.apply(LabelEncoder().fit_transform), continuous_values], axis=1)
X = data.as_matrix()
y = credit_values.as_matrix()

cost = np.vectorize(lambda t: 1 if t == '1'else 5)

X_scaled = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
print "Scale Score apprentissage = %f" % clf.score(X_train, y_train)
print "Scale Score test = %f" % clf.score(X_test, y_test)


### MIN MAX
X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
print "MinMax Score apprentissage = %f" % clf.score(X_train, y_train)
print "MinMax Score test = %f" % clf.score(X_test, y_test)

### Std SCaler
X_scaled = preprocessing.StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
print "Std Score apprentissage = %f" % clf.score(X_train, y_train)
print "Std Score test = %f" % clf.score(X_test, y_test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
print "No Score apprentissage = %f" % clf.score(X_train, y_train)
print "No Score test = %f" % clf.score(X_test, y_test)