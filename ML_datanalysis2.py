# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 11:07:46 2018

@author: olhartin@asu.edu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import cm as cm
from sklearn.decomposition import PCA
import seaborn as sns
## create covariance for dataframes
def mosthighlycorrelated(mydataframe, numtoreport): 
# find the correlations 
    cormatrix = mydataframe.corr() 
# set the correlations on the diagonal or lower triangle to zero, 
# so they will not be reported as the highest ones: 
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T 
# find the top n correlations 
    cormatrix = cormatrix.stack() 
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index() 
# assign human-friendly names 
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"] 
    return cormatrix.head(numtoreport)
## Covariance matrix
def correl_matrix(X,cols):
    fig = plt.figure(figsize=(7,7), dpi=100)
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet',30)
    cax = ax1.imshow(np.abs(X.corr()),interpolation='nearest',cmap=cmap)
##    ax1.set_xticks(major_ticks)
    major_ticks = np.arange(0,len(cols),1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.grid(True,which='both',axis='both')
##    plt.aspect('equal')
    plt.title('Correlation Matrix')
    labels = cols
    ax1.set_xticklabels(labels,fontsize=9)
    ax1.set_yticklabels(labels,fontsize=12)
    fig.colorbar(cax, ticks=[-0.4,-0.25,-.1,0,0.1,.25,.5,.75,1])
    plt.show()
    return(1)
## make pair plots
def pairplotting(df):
    sns.set(style='whitegrid', context='notebook')
    cols = df.columns
##bcols = ['age', 'sex', 'cpt', 'rbp', 'sc', 'fbs', 'rer', 'mhr', 'eia', 'opst', 'dests', 'nmvcf', 'thal', 'a1p2']
    sns.pairplot(df[cols],size=2.5)
    plt.show()
## this creates a dataframe similar to a dictionary
## a data frame can be constructed from a dictionary
## https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
iris = pd.read_csv('IRIS2.csv')
print('first 5 observations',iris.head(5))
cols = iris.columns
X = iris.iloc[:,0:3].values
Y = iris.iloc[:,4].values
## Identify Null values
print(' Identify Null Values ')
print( iris.apply(lambda x: sum(x.isnull()),axis=0) )
## 'setosa' 0
## 'versicolor' 1
## 'virginica' 2
##  descriptive statistics
print(' Descriptive Statistics ')
print(iris.describe())
## most highly correlated lists
print("Most Highly Correlated")
print(mosthighlycorrelated(iris,5))
## heat plot of covariance
print(' Covariance Matrix ')
correl_matrix(iris.iloc[:,0:5],cols[0:5])
## Pair plotting
print(' Pair plotting ')
pairplotting(iris)
"""
"""