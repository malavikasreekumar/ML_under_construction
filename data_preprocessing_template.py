# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 11:07:25 2018

@author: Malavika  Sreekumar
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#importing datasets
dataset= pd.read_csv('data1.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 3].values

# Splitting test set and training set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,train_size=.75,random_state=0)#the train size and test size tells us the part of dataset we take for each


#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)"""