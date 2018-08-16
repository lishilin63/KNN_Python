#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 13:15:42 2018

@author: shilinli
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

dt = pd.read_csv('KNN_Project_Data')

# sb.pairplot(dt)

# Standardize the Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dt.drop('TARGET CLASS',axis = 1))
scaled_dt = pd.DataFrame(scaler.transform(dt.drop('TARGET CLASS',axis = 1)),columns = dt.columns[:-1])

# Train Test Split
from sklearn.cross_validation import train_test_split
x = scaled_dt
y = dt['TARGET CLASS']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 101 )

# Using KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)

# Predictions and Evaluations
pred = knn.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

# Choosing a K Value
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.plot(range(1,40),error_rate,color = 'blue',linestyle = 'dashed',marker = 'o',markerfacecolor = 'red',markersize = 5)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# Choose K = 31
knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(x_train,y_train) 
pred = knn.predict(x_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
