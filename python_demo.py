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
chdir('Documents/Stats/Repositories/Docs/introduction-to-machine-learning')
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


import tensorflow.contrib.learn as skflow
from sklearn import metrics

y = wine['good'] == 'Good'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8675309)


feats = skflow.infer_real_valued_columns_from_input(X_train)

classifier_tf = skflow.DNNClassifier(feature_columns=feats, 
                                     hidden_units=[50, 50, 50, 40, 30, 20, 10], 
                                     dropout=.2,
                                     n_classes=2)
classifier_tf.fit(X_train, y_train, steps=10000)
predictions = list(classifier_tf.predict(X_test, as_iterable=True))
score = metrics.accuracy_score(y_test, predictions)
print("Accuracy: %f" % score)