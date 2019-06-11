# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:56:57 2019

@author: Malavika  Sreekumar
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#Using Dentogram to find optimal no of clusters
import scipy.cluster.hierarchy as sch
dentogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.titile('Dentogram')
plt.xlabel('Customers')
plt.ylabel('Eucliean distance')
plt.show()

#Fittitng Hierrchical cluster to dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)

#Visualizing Clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='Careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='Standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='Targets')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='Careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='magenta',label='Sensible')

plt.title('Clusters of clients')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending score(1-100)')
plt.legend()
plt.show()