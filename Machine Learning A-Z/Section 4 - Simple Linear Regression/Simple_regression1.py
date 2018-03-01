# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 23:47:10 2017

@author: kd
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
datasets = pd.read_csv('Salary_Data.csv')
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, 1].values



# Lets split the datasets into test sets and training sets
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.333, random_state = 0)

# Featrues Scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""



# Fitting Simple linear regression to the Traning set
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)

# Predicting te test set result
Y_pred = regression.predict(X_test)

# Visualising the Training set result
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regression.predict(X_train), color = 'brown')
plt.title('Salary vs Exprience (Traininh Sets)')
plt.xlabel('Years, of Exprience')
plt.ylable('Salary')


# Visualising the test set result
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regression.predict(X_train), color = 'blue')
plt.title('Salary vs Exprience (Test    Sets)')
plt.xlabel('Years, of Exprience')
plt.ylable('Salary')     