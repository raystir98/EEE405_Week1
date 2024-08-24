# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 11:13:49 2021

@author: olhartin@asu.edu
https://stackoverflow.com/questions/22956938/exhaustively-feature-selection-in-scikit-learn
"""
##
## This approach does an exhuastive search of feature combinations and 
##  reports on the best of each 
##  abitrary classifiers may be used
##
##  https://docs.python.org/3/library/itertools.html
from itertools import chain, combinations
##  https://scikit-learn.org/0.15/modules/generated/sklearn.cross_validation.cross_val_score.html
from sklearn.model_selection import cross_val_score
import numpy as np

##  Calculates the best model of up to max_size features of X.
##   estimator must have a fit and score functions.
##   X must be a DataFrame.
##  cv is the number of cross validation folds, 
##  larger cv reduces the amount of data in each training so there may be warnings
##
def best_subset(estimator, X, y, max_size=8, cv=5):

    n_features = X.shape[1]
##      creates every combination of features
    subsets = (combinations(range(n_features), k + 1) 
               for k in range(min(n_features, max_size)))
##      finds best subsets of each given length
    best_size_subset = []
    for subsets_k in subsets:  # for each list of subsets of the same size
        best_score = -np.inf
        best_subset = None
        for subset in subsets_k: # for each subset
            estimator.fit(X.iloc[:, list(subset)], y)
            # get the subset with the best score among subsets of the same size
            score = estimator.score(X.iloc[:, list(subset)], y)
            if score > best_score:
                best_score, best_subset = score, subset
        # to compare subsets of different sizes we must use CV
        # first store the best subset of each size
        best_size_subset.append(best_subset)
##
##      compare best subsets of each size determining their scores
##
    best_score = -np.inf
    best_subset = None
    list_scores = []
    for subset in best_size_subset:
        score = cross_val_score(estimator, X.iloc[:, list(subset)], y, cv=cv).mean()
        list_scores.append(score)
        print(score)
        print(subset)
        if score > best_score:
            best_score, best_subset = score, subset


    return best_subset, best_score, best_size_subset, list_scores
##
##  read dataset into pandas dataframe
##
import pandas as pd
iris = pd.read_csv('IRIS2.csv')
print('first 5 observations',iris.head(5))
cols = iris.columns
X = iris.iloc[:,0:4].values
Y = iris.iloc[:,4].values
##
##  one hot version of target Y
##
from sklearn import preprocessing
y = preprocessing.label_binarize(Y, classes=[0,1,2])
##
##  Normalize data
##
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)
##
#from sklearn.ensemble import RandomForestClassifier
#bs = best_subset(RandomForestClassifier(), iris.iloc[:,0:4], y, max_size=4, cv=5)
##
from sklearn.linear_model import LogisticRegression
bs = best_subset(LogisticRegression(solver='lbfgs'), iris.iloc[:,0:4], Y, max_size=4, cv=2)
##
##  print best performance using k fold cross covariance
##
print('best subset ', bs[0])
print('best score ', bs[1])
print('best size subset ', bs[2])
print('best list scores ', bs[3])