# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 13:42:26 2017

@author: kd
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets

X = datasets.iloc[:, 1:2].values
Y = datasets.iloc[:, 2].values
 

# Lets split the datasets into test sets and training sets
"""from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)"""

# Featrues Scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression model to the datasets
from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

#Fitting ploynomial regression "To The Dataseta
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, Y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualzing the linear Regression Results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title("truth or Bluff (linear Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')

# visualzing Polynomial Regression Result
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title("truth or Bluff (Polynomial Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting new result with linear Regression
lin_reg.predict(6.5)

# Predicting new with Polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))





