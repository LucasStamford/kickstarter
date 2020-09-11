# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 00:13:07 2020
@author: Stamford
https://www.kaggle.com/kemical/kickstarter-projects
"""

import os
os.chdir('F:\DataSets\Kickstarter')

import pandas as pd

ks = pd.read_csv('ks-projects-201801.csv')

ks = ks[ks['state'] != 'live']
ks = ks[ks['state'] != 'undefined']
ks['state'] = ks['state'].str.replace('canceled|suspended', 'failed')
ks['deadline'] = pd.to_datetime(ks['deadline'])
ks['launched'] = pd.to_datetime(ks['launched'])

ks.insert(6,'time',(ks['deadline'].dt.date - ks['launched'].dt.date).dt.days)

ks.insert(7, 'year', ks['launched'].dt.year)
ks.insert(8, 'month', ks['launched'].dt.month)
ks.insert(9, 'day', ks['launched'].dt.day)
ks.insert(9, 'weekday', ks['launched'].dt.weekday)

ks.insert(10, 'num_chars', ks['name'].apply(lambda x : len(str(x).replace(' ', ''))))

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
ks['category'] = label.fit_transform(ks['category'])
ks['main_category'] = label.fit_transform(ks['main_category'])
ks['state'] = label.fit_transform(ks['state'])
ks['country'] = label.fit_transform(ks['country'])
ks['currency'] = label.fit_transform(ks['currency'])

X = ks[['category', 'main_category', 'time', 'year', 'month', 'day', 'weekday', 'num_chars', 'currency', 'country', 'usd_goal_real']]
y = ks['state']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3, stratify=y)

import lightgbm as lgb
gbm = lgb.LGBMRegressor(num_leaves=128, metric='auc', objective='binary')
gbm.fit(X_train, y_train)
y_predict = gbm.predict(X_test)

#y_predict = (y_predict > 0.5).astype(int)

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
auc = roc_auc_score(y_test, y_predict)
#matrix = confusion_matrix(y_test, y_predict)
#accuracy = accuracy_score(y_test, y_predict)
