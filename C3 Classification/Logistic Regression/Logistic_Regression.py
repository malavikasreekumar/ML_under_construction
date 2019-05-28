# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:07:54 2019

@author: Malavika  Sreekumar
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#importing datasets
dataset= pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:, 4].values


#Splitting the data set into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.25, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)  

#Fitting Logistic Regression to Training Set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#Predicting test results
y_pred=classifier.predict(X_test)

#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)