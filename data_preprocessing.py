#Data preprocessing

#importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#importing datasets
dataset= pd.read_csv('data1.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X= LabelEncoder()
X[:,0]= labelencoder_X.fit_transform(X[:,0])
onehotencoder_X=OneHotEncoder(categorical_features=[0])
X=onehotencoder_X.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Splitting test set and training set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,train_size=.75,random_state=0)#the train size and test size tells us the part of dataset we take for each

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

#It is not necessary to fit and transform dummy variables,it depends on our data

#We do not need to apply feature scaling to dependent variable 'y' as in this case it is a classification problem,but
# for regression problems we need to feature scale 