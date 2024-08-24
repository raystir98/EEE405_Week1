# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 17:09:35 2020

@author: olhartin@asu.edu
"""

# evaluate RFE for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# define dataset
import pandas as pd
iris = pd.read_csv('IRIS2.csv')
print('first 5 observations',iris.head(5))
cols = iris.columns
X = iris.iloc[:,0:4].values
Y = iris.iloc[:,4].values
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)
# create pipeline
#rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=2)
rfe = RFE(estimator=LogisticRegression(solver='lbfgs'), n_features_to_select=2)
fit = rfe.fit(X_std, Y)
print("Number of Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
#model = DecisionTreeClassifier()
model = LogisticRegression(solver='lbfgs')
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X_std, Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))