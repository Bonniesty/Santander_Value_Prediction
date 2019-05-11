#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:20:09 2019

@author: tysun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import lightgbm as lgb
import xgboost as xgb
import random
random.seed(2018)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


#1.First step, we load train and test data
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
X_train = train_df.drop(["ID", "target"], axis=1)
y_train = np.log1p(train_df["target"].values)
X_test = test_df.drop(["ID"], axis=1)



#2. Then we remove some ocnstan and duplicate data
#remove constant
colsToRemove = []
for col in X_train.columns:
    if X_train[col].std() == 0: 
        colsToRemove.append(col)
        
# remove constant columns in the training set
X_train.drop(colsToRemove, axis=1, inplace=True)

# remove constant columns in the test set
X_test.drop(colsToRemove, axis=1, inplace=True) 

print("Removed `{}` Constant Columns\n".format(len(colsToRemove)))
print(colsToRemove)

#remove duplicate
colsToRemove = []
colsScaned = []
dupList = {}
columns = X_train.columns
for i in range(len(columns)-1):
    v = X_train[columns[i]].values
    dupCols = []
    for j in range(i+1,len(columns)):
        print(1)
        if np.array_equal(v, X_train[columns[j]].values):
            print(2)
            colsToRemove.append(columns[j])
            if columns[j] not in colsScaned:
                dupCols.append(columns[j]) 
                colsScaned.append(columns[j])
                dupList[columns[i]] = dupCols
                
X_train.drop(colsToRemove, axis=1, inplace=True) 
X_test.drop(colsToRemove, axis=1, inplace=True)

print("After remove Constant and Dup Train set size: {}".format(X_train.shape))
print("After remove Constant and Dup Test set size: {}".format(X_test.shape))


#plot varience

standardized_train = StandardScaler().fit_transform(X_train.values)
tot = sum(eig_vals)

# Individual explained variance
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] 
var_exp_real = [v.real for v in var_exp]

# Cumulative explained variance
cum_var_exp = np.cumsum(var_exp)
mean_vec = np.mean(standardized_train, axis=0)
cov_matrix = np.cov(standardized_train.T)
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the eigenvalue, eigenvector pair from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

# Calculation of Explained Variance fro
cum_exp_real = [v.real for v in cum_var_exp]



import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

trace1 = go.Scatter(x=X_train.columns, y=var_exp_real, name="Individual Variance", opacity=0.75, marker=dict(color="red"))
trace2 = go.Scatter(x=X_train.columns, y=cum_exp_real, name="Cumulative Variance", opacity=0.75, marker=dict(color="black"))
layout = dict(height=400, title='Variance Explained by Variables', legend=dict(orientation="h", x=0, y=1.2));
fig = go.Figure(data=[trace1, trace2], layout=layout);

iplot(fig);

#3  Thenm we perform fearture learning

def addSumZeros(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','target']]
    if 'SumZeros' in features:
        train.insert(1, 'SumZeros', (train[flist] == 0).astype(int).sum(axis=1))
        test.insert(1, 'SumZeros', (test[flist] == 0).astype(int).sum(axis=1))
    flist = [x for x in train.columns if not x in ['ID','target']]
    return train, test

def addSumValues(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','target']]
    if 'SumValues' in features:
        train.insert(1, 'SumValues', (train[flist] != 0).astype(int).sum(axis=1))
        test.insert(1, 'SumValues', (test[flist] != 0).astype(int).sum(axis=1))
    flist = [x for x in train.columns if not x in ['ID','target']]

    return train, test

def addOtherAgg(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','target','SumZeros','SumValues']]
    if 'OtherAgg' in features:
        train['Mean']   = train[flist].mean(axis=1)
        train['Median'] = train[flist].median(axis=1)
        train['Mode']   = train[flist].mode(axis=1)
        train['Max']    = train[flist].max(axis=1)
        train['Var']    = train[flist].var(axis=1)
        train['Std']    = train[flist].std(axis=1)
        
        test['Mean']   = test[flist].mean(axis=1)
        test['Median'] = test[flist].median(axis=1)
        test['Mode']   = test[flist].mode(axis=1)
        test['Max']    = test[flist].max(axis=1)
        test['Var']    = test[flist].var(axis=1)
        test['Std']    = test[flist].std(axis=1)
    flist = [x for x in train.columns if not x in ['ID','target','SumZeros','SumValues']]

    return train, test

X_train, X_test = addSumZeros(X_train, X_test, ['SumZeros'])
X_train, X_test = addSumValues(X_train, X_test, ['SumValues'])
X_train, X_test = addOtherAgg(X_train, X_test, ['OtherAgg'])


#Perform k means
flist_kmeans = []
flist = [x for x in X_train.columns if not x in ['ID','target']]
for ncl in range(2,11):
    cls = KMeans(n_clusters=ncl)
    cls.fit_predict(X_train[flist].values)
    X_train['kmeans_cluster_'+str(ncl)] = cls.predict(X_train[flist].values)
    X_test['kmeans_cluster_'+str(ncl)] = cls.predict(X_test[flist].values)


#Perform PCA
n_components = 20
flist_pca = []
pca = PCA(n_components=n_components)
flist = [x for x in X_train.columns if not x in ['ID','target']]
x_train_projected = pca.fit_transform(normalize(X_train[flist], axis=0))
x_test_projected = pca.transform(normalize(X_test[flist], axis=0))
for npca in range(0, n_components):
    X_train.insert(1, 'PCA_'+str(npca+1), x_train_projected[:, npca])
    X_test.insert(1, 'PCA_'+str(npca+1), x_test_projected[:, npca])

def plot_3_components(x_trans, title):
    trace = go.Scatter3d(x=x_trans[:,0], y=x_trans[:,1], z = x_trans[:,2],
                          mode = 'markers', showlegend = False,
                          marker = dict(size = 8, color=x_trans[:,1], 
                          line = dict(width = 1, color = '#f7f4f4'), opacity = 0.5))
    layout = go.Layout(title = title, showlegend= True)
    fig = dict(data=[trace], layout=layout)
    iplot(fig)
plot_3_components(x_train_projected, 'First Three Component of PCA')
