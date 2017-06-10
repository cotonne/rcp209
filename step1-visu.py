#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import genfromtxt

import csv
import collections
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2 as chi2_s
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn import tree

sns.set(style="whitegrid", color_codes=True)


# Reading CSV
filename = 'german.data.csv'
delimiter = ' '
data = []

continuous_values = pd.read_csv(filename, delimiter=delimiter, 
    names=['Duration in month', 'Credit amount', 'Installement rate', 'Present residence since', 'Age', 'Number of existing credits', 'Nb of liable people'],
    usecols=[1,4, 7, 10, 12, 15, 17],
    dtype='float')


values = continuous_values.columns.values

# Decile
print np.percentile(continuous_values, np.arange(0, 100, 10), axis=0)

# Affiche les valeurs sous forme de distribution
f, axarr = plt.subplots(4, 2)
for index in range(0, len(values)):
  print 'Reading ' + values[index]
  sns.distplot(continuous_values[values[index]], ax=axarr[index/2, index % 2])

f, axarr = plt.subplots(4, 2)
for index in range(0, len(values)):
  print 'Reading ' + values[index]
  sns.boxplot(continuous_values[values[index]], ax=axarr[index/2, index % 2])




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




#####
# Calcul des tables de contingences
for index in discrete_values.columns:
  print pd.crosstab(credit_values, discrete_values[index], margins=True)

unique, counts = np.unique(discrete_values, return_counts=True)
dict(zip(unique, counts))

values = discrete_values.columns.values

# Affiche les valeurs sous forme de distribution
f, axarr = plt.subplots(5, 3, figsize=(10, 10))
for index in range(0, len(values)):
  print 'Reading ' + values[index]
  sns.countplot(x=values[index], data=discrete_values, 
    ax=axarr[index/3, index % 3]);
plt.savefig("continuous1.png")

with_credit = discrete_values.copy()
with_credit["Credit"] = credit_values
f, axarr = plt.subplots(5, 3, figsize=(10, 10))
for index in range(0, len(values)):
  print 'Reading ' + values[index]
  ax = sns.countplot(x=values[index], hue='Credit', data=with_credit,
    ax=axarr[index/3, index % 3]);

plt.savefig("continuous2.png")

f, axarr = plt.subplots(6, 5, figsize=(10, 10))
for index in range(0, len(values)):
  print 'Reading ' + values[index]
  datax = discrete_values[values[index]]
  pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, ax=axarr[index/5, index % 5])

plt.savefig("continuous3.png")

##### X²
# The p-value of a feature selection score indicates the probability 
# that this score or a higher score would be obtained if this variable 
# showed no interaction with the target.
#  scores are better if greater, p-values are better if smaller (and losses are better if smaller)
# Another general statement: scores are better if greater, p-values are better if smaller (and losses are better if smaller)
cv = pd.DataFrame(MinMaxScaler().fit_transform(continuous_values), columns=continuous_values.columns)
X = pd.concat([discrete_values.apply(LabelEncoder().fit_transform), cv], axis=1).as_matrix()
y = credit_values.as_matrix()
## on garde les 11 meilleurs features
chi2, pvalues  =  chi2_s(X, y)

print "chi2"
print chi2

featureSelector = SelectKBest(score_func=chi2_s, k=11)
featureSelector.fit_transform(X, y)


###############################################################
## AFCM
values = continuous_values.columns.values

dummies = pd.DataFrame()
dummies['Duration in month'] = pd.qcut(continuous_values['Duration in month'], 7, labels=False)
dummies['Credit amount'] = pd.qcut(continuous_values['Credit amount'], 10, labels=False)
dummies['Age'] = pd.qcut(continuous_values['Age'], 10, labels=False)

data = pd.concat([discrete_values.apply(LabelEncoder().fit_transform), dummies,
  continuous_values['Present residence since'],
  continuous_values['Installement rate'],
  continuous_values['Number of existing credits'],
  continuous_values['Nb of liable people']], axis=1)


fa = FactorAnalysis()
X = data.as_matrix()
y = credit_values.as_matrix()
X_transformed = fa.fit_transform(X, y)


######################## 
## correlation
dummies = pd.get_dummies(discrete_values.ix[:,:'Foreigner'], columns=['Status of checking account', 'Credit history', 
         'Purpose', 'Savings account', 'Present employment since', 
         'Personal status and sex', 'Guarantors', 'Property',
         'Other installment plans', 'Housing', 
         'Job', 'Telephone', 'Foreigner'], drop_first = True)



data = pd.concat([continuous_values, dummies], axis=1)


covariance_matrix = np.corrcoef(data.transpose())

### Correlations
df = data.corr()
labels = df.where(np.triu(np.ones(df.shape)).astype(np.bool))
labels = labels.round(2)
labels = labels.replace(np.nan,' ', regex=True)

mask = np.triu(np.ones(df.shape)).astype(np.bool)
ax = sns.heatmap(df, mask=mask, cmap='RdYlGn_r', fmt='', square=True, linewidths=1.5)
mask = np.ones((48, 48))-mask
ax = sns.heatmap(df, mask=mask, cmap=ListedColormap(['white']),annot=labels,cbar=False, fmt='', linewidths=1.5)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("correlation.png")
plt.matshow(covariance_matrix, cmap=plt.cm.gray)


##g = sns.PairGrid(data)
#g.map_diag(sns.kdeplot)
#g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6);


#sns.pairplot(data);


# Minimum percentage of variance we want to be described by the resulting transformed components
variance_pct = .99

# Create PCA object
pca = PCA(n_components=variance_pct)

# Transform the initial features
X_transformed = pca.fit_transform(X,y)

plt.plot(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_)

# Create a data frame from the PCA'd data
pcaDataFrame = pd.DataFrame(X_transformed)

print pcaDataFrame.shape[1], " components describe ", str(variance_pct)[1:], "% of the variance"

plt.show()
