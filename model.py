#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 1 10:42:48 2019

    @author: Tianyi Sun, Yueling Jiang, Yang Qiao
"""

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import lightgbm as lgb
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
random.seed(2018)

#1.First step, we load train and test data
trainRaw = pd.read_csv('./data/train.csv')
testRaw = pd.read_csv('./data/test.csv')
train = trainRaw.drop(["ID", "target"], axis=1)
target = np.log1p(trainRaw["target"].values)
test = testRaw.drop(["ID"], axis=1)

#2. Then we remove some constant and duplicate data
#remove constant
colsToRemove = []
for col in train.columns:
    if train[col].std() == 0: 
        colsToRemove.append(col)

train.drop(colsToRemove, axis=1, inplace=True)
test.drop(colsToRemove, axis=1, inplace=True) 

#remove duplicate
colsToRemove = []
colsScaned = []
dupList = {}
columns = train.columns
for i in range(len(columns)-1):
    v = train[columns[i]].values
    dupCols = []
    for j in range(i+1,len(columns)):
        if np.array_equal(v, train[columns[j]].values):
            colsToRemove.append(columns[j])
            if columns[j] not in colsScaned:
                dupCols.append(columns[j]) 
                colsScaned.append(columns[j])
                dupList[columns[i]] = dupCols                
train.drop(colsToRemove, axis=1, inplace=True) 
test.drop(colsToRemove, axis=1, inplace=True)

print("After remove Constant and Dup Train set size: {}".format(train.shape))
print("After remove Constant and Dup Test set size: {}".format(test.shape))

#3  Then we perform fearture learning

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

train, test = addSumZeros(train, test, ['SumZeros'])
train, test = addSumValues(train, test, ['SumValues'])
train, test = addOtherAgg(train, test, ['OtherAgg'])


#Draw variance plot
#The following part need to be run in notebook!!! 
'''
#plot for variance
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from sklearn.preprocessing import StandardScaler
init_notebook_mode(connected=True)
# Calculating Eigenvectors and eigenvalues
standardized_train = StandardScaler().fit_transform(train.values)
mean_vec = np.mean(standardized_train, axis=0)
cov_matrix = np.cov(standardized_train.T)
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort(key = lambda x: x[0], reverse= True)
tot = sum(eig_vals)
# Individual explained variance
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] 
var_exp_real = [v.real for v in var_exp]
# Cumulative explained variance
cum_var_exp = np.cumsum(var_exp) 
cum_exp_real = [v.real for v in cum_var_exp]

trace1 = go.Scatter(x=train.columns, y=var_exp_real, name="Individual Variance", opacity=0.75, marker=dict(color="red"))
trace2 = go.Scatter(x=train.columns, y=cum_exp_real, name="Cumulative Variance", opacity=0.75, marker=dict(color="black"))
layout = dict(height=400, title='Variance Explained by Variables', legend=dict(orientation="h", x=0, y=1.2));
fig = go.Figure(data=[trace1, trace2], layout=layout);
iplot(fig);
'''
#Perform k means
flist_kmeans = []
flist = [x for x in train.columns if not x in ['ID','target']]
for ncl in range(2,11):
    cls = KMeans(n_clusters=ncl)
    cls.fit_predict(train[flist].values)
    train['kmeans_cluster_'+str(ncl)] = cls.predict(train[flist].values)
    test['kmeans_cluster_'+str(ncl)] = cls.predict(test[flist].values)


#Perform PCA
n_components = 20
flist_pca = []
pca = PCA(n_components=n_components)
flist = [x for x in train.columns if not x in ['ID','target']]
train_projected = pca.fit_transform(normalize(train[flist], axis=0))
test_projected = pca.transform(normalize(test[flist], axis=0))
for npca in range(0, n_components):
    train.insert(1, 'PCA_'+str(npca+1), train_projected[:, npca])
    test.insert(1, 'PCA_'+str(npca+1), test_projected[:, npca])

#Draw PCA plot
#The following part need to be run in notebook!!! 
'''
#draw PCA plot
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
def plot_3_components(x_trans, title):
    trace = go.Scatter3d(x=x_trans[:,0], y=x_trans[:,1], z = x_trans[:,2],
                          mode = 'markers', showlegend = False,
                          marker = dict(size = 8, color=x_trans[:,1], 
                          line = dict(width = 1, color = '#f7f4f4'), opacity = 0.5))
    layout = go.Layout(title = title, showlegend= True)
    fig = dict(data=[trace], layout=layout)
    iplot(fig)
plot_3_components(train_projected, 'First Three Component of PCA')
'''

#4 Last step, run model
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "learning_rate" : 0.01,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgtrain, lgval], early_stopping_rounds=100, 
                      verbose_eval=200, evals_result=evals_result)    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result

# Training LGB
seeds = [42, 2018]
pred_test_full_seed = 0
for seed in seeds:
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
    pred_test_full = 0
    for dev_index, val_index in kf.split(train):
        dev_X, val_X = train.loc[dev_index,:], train.loc[val_index,:]
        dev_y, val_y = target[dev_index], target[val_index]
        pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test)
        pred_test_full += pred_test
    pred_test_full /= 5.
    pred_test_full = np.expm1(pred_test_full)
    pred_test_full_seed += pred_test_full
pred_test_full_seed /= np.float(len(seeds))
print("LightGBM Training Completed!!!")


# feature importance
print("Features Importance")
gain = model.feature_importance('gain')
featureimp = pd.DataFrame({'feature':model.feature_name(), 
                   'split':model.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)

#5 Draw Top 15 important features and write to submission
print(featureimp[:15])
_ = lgb.plot_metric(evals_result)

# write to submission
sub = pd.read_csv('./data/sample_submission.csv')
sub["target"] = pred_test_full_seed
sub.to_csv('submission.csv', index=False)
