# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:03:15 2019

@author: Malavika  Sreekumar
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Fitting linear Regression to data model
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#Fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualising the Linear Regression result
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Positional level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression result
X_grid=np.arange(min(X),max(X),0.1)#This will give us a better curve ie higher resolution
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Positional level')
plt.ylabel('Salary')
plt.show()

#Predicting  a new result with Linear Regression
X_grid=X_grid.reshape((len(X_grid),1))
lin_reg.predict(6.5)
 