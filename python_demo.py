# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 11:00:14 2017

@author: MC
"""

import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from os import chdir

# import data
chdir('D:\Documents\Stats\Repositories\Docs\introduction-to-machine-learning')
wine = pd.read_csv('data/wine.csv')


# data preprocessing
np.random.seed(1234)
X = wine.drop(['free.sulfur.dioxide', 'density', 'quality', 'color', 'white','good'], axis=1)
X = MinMaxScaler().fit_transform(X)  # by default on 0, 1 scale
y = wine['good']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# train model
rf = RandomForestClassifier(n_estimators=1000)
rf_train = rf.fit(X_train, y_train)


# get test predictions
rf_predict = rf_train.predict(X_test)

# create confusion matrix, and accuracy
cm = sk.metrics.confusion_matrix(y_test,rf_predict)
cm_prob = cm / np.sum(cm)
cm_prob 

acc = sk.metrics.accuracy_score(y_test, rf_predict)
acc = pd.DataFrame(np.array([acc]), columns=['Accuracy'])
acc