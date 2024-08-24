# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 20:45:23 2016

@author: olhartin@asu.edu
"""
## Linear Regression
## use regression to determine how well
## General electric investment impacted 
## Shares outstanding and share price (or market cap)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm as cm
import seaborn as sns   ## conda install seaborn
##
##  correl matrix
##
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
##
##      make pair plots
##
def pairplotting(df):
    sns.set(style='whitegrid', context='notebook')
    cols = df.columns
##bcols = ['age', 'sex', 'cpt', 'rbp', 'sc', 'fbs', 'rer', 'mhr', 'eia', 'opst', 'dests', 'nmvcf', 'thal', 'a1p2']
    sns.pairplot(df[cols],size=2.5)
    plt.show()
##
##      read csv using pandas library
##   
import pandas as pd
df = pd.read_csv('sasch3p40.csv')  ## returns dataframe df
y = df['I'].values                  ## column 'I' of dataframe containing target which is col 1
Size = len(y)                       ## Size is the number of observations
X = np.hstack((df.iloc[:,0].values.reshape(Size,1),df.iloc[:,2:4].values))
#X = df.iloc[:,2:4].values      ##  not including Year
cols = df.columns
##
##      correlation matrix
##
print(' Covariance Matrix ')
correl_matrix(df.iloc[:,0:5],cols[0:5])
##
##      Pair plotting
##
print(' Pair plotting ')
pairplotting(df)
##
##  Year
##  I gross investment of GE General Electric
##  C Capital stock Lagged GE (ooutstanding shares)
##  F Value of shares GE Lagged (price)
##
# plt.scatter(X[:,0],y,c='blue',marker='o')
# plt.scatter(X[:,1],y,c='red',marker='+')
# plt.scatter(X[:,2],y,c='g',marker='s')      ##  Year
# plt.grid()
# plt.show()
##
##      conda install -c anaconda scikit-learn
##
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
##
##      standard scalar
##
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
##
##      scaled data
##
plt.scatter(range(0,len(X_train_std[:,0])),X_train_std[:,0],c='r')
plt.scatter(range(0,len(X_train_std[:,1])),X_train_std[:,1],c='b')
plt.scatter(range(0,len(X_train_std[:,2])),X_train_std[:,2],c='g') ## Year
plt.title('data')
plt.xlabel('index')
plt.ylabel('value')
plt.grid()
plt.show()
##
##      Least Squares Solution, compact notation
##      Direct solve
##
# Size = len(X_train_std[:,0])
# Ones = np.ones(Size,dtype=float).reshape(Size,1)  ## make a column vector
# A = np.hstack((Ones,X_train_std))

Size = len(X_train_std[:,0])    ## generally compact notation isn't used in code
Ones =  np.ones(Size)           ## but it is interesting for this problem
A = np.column_stack((Ones,X_train_std))

ATA = np.matmul(np.transpose(A),A)
ATY = np.matmul(np.transpose(A),y_train)
##
##      solve for weights through inversion
##
W = np.matmul(np.linalg.inv(ATA),ATY)
##
##      Error prediction
##
print('\n Our Least Squares Code \n')
y_LS_pred = np.matmul(A,W)
from sklearn.metrics import r2_score
R2_LS = r2_score(y_train,y_LS_pred)
print(' training Weights ', W, '\n R2 ', R2_LS)
##
##      Method 2 solve using solve, more stable
##  numpy.linalg.solve — NumPy v1.18 Manual
##
W = np.linalg.solve(ATA,ATY)
##
##      Error prediction
##
y_LS_pred = np.matmul(A,W)
from sklearn.metrics import r2_score
R2_LS = r2_score(y_train,y_LS_pred)
print(' solve Weights ', W, ' R2 ', R2_LS)
##
##  test
##
Size = len(X_test_std[:,0])
Ones = np.ones(Size,dtype=float).reshape(Size,1)  ## make a column vector
A = np.hstack((Ones,X_test_std))
ATA = np.matmul(np.transpose(A),A)
ATY = np.matmul(np.transpose(A),y_test)
y_LS_pred = np.matmul(A,W)
from sklearn.metrics import r2_score
R2_LS = r2_score(y_test,y_LS_pred)
print(' test R2 ', R2_LS)
##
##      Method 3 Linear Regression library
##
print('\n Sklearn Least Squares Code \n')
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X_train_std,y_train)
y_train_pred = slr.predict(X_train_std)
y_test_pred = slr.predict(X_test_std)
print(' coefficients ', slr.coef_,' intercept ', slr.intercept_)
R2_LS = r2_score(y_train,y_train_pred)
print(' R2 train ', R2_LS)
R2_LS = r2_score(y_test,y_test_pred)
print(' R2 test ', R2_LS)

##
##      Residuals
##
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.grid()
plt.show()
## plt.hlines(y=0,xmin=-10,xmax=50,lw=2,color='red')
## plt.xlim([-10,50])

y_combined = np.hstack((y_train, y_test))
y_combined_pred = np.hstack((y_train_pred, y_test_pred))

##
##      Print our errors
##
from sklearn.metrics import r2_score
R2_train = r2_score(y_train,y_train_pred)
R2_test =  r2_score(y_test,y_test_pred)
R2_combined = r2_score(y_combined, y_combined_pred)

print(' R2 train %.3f ' % R2_train )
print(' R2 test %.3f ' % R2_test )
print(' R2 combined: %.3f ' % R2_combined )

if (R2_test>R2_train):
    print(' Overfitting? R2_test %.3f ' % R2_test, ' < R2_train %.3f ' % R2_train )
else:
    print(' Underfit? ')
##
##  performance
##
plt.scatter(y_train,y_train_pred)
plt.plot(y_train,y_train)
plt.xlabel('targets')
plt.ylabel('predicted targets')
plt.title('performance')
plt.grid()
plt.grid()
plt.show()
