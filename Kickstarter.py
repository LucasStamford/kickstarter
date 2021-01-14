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

from sklearn.model_selection import GridSearchCV

models = []
name_models = []
roc_auc = []

from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(X_train, y_train)

name_models.append(type(naive).__name__)
models.append(naive)


from sklearn.linear_model import LinearRegression
linear = LinearRegression(n_jobs=-1)
linear.fit(X_train, y_train)

name_models.append(type(linear).__name__)
models.append(linear)


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
param_grid = {
    'criterion':['gini','entropy'],
    'max_depth':[4,5,6,8,10,20,40,80,100]
    }
grid = GridSearchCV(tree, param_grid=param_grid, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

name_models.append(type(tree).__name__)
models.append(grid)


from sklearn.svm import SVC
svc = SVC()
param_grid = {
    'C':[1,10,100,1000],
    'gamma':[1,0.1,0.001,0.0001], 
    'kernel':['linear','rbf']
    }
grid = GridSearchCV(svc, param_grid=param_grid, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

name_models.append(type(svc).__name__)
models.append(grid)


from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier()
param_grid = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    }
grid = GridSearchCV(random, param_grid=param_grid, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

name_models.append(type(random).__name__)
models.append(grid)


import xgboost
xgb = xgboost.XGBClassifier(objective='binary:logistic')
param_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
grid = GridSearchCV(xgb, param_grid=param_grid, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

name_models.append(type(xgb).__name__)
models.append(grid)


import lightgbm as lgb
gbm = lgb.LGBMRegressor(metric='auc', objective='binary', verbose=1)
param_grid = {
    'num_leaves': [64, 128, 256],
    'reg_alpha': [0.1, 0.5],
    'min_data_in_leaf': [50, 100, 300, 400],
    'lambda_l1': [0, 1, 1.5],
    'lambda_l2': [0, 1]
    }
grid = GridSearchCV(gbm, param_grid=param_grid, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

name_models.append(type(gbm).__name__)
models.append(grid)

#y_predict = (y_predict > 0.5).astype(int)

from sklearn.metrics import roc_auc_score
for model in models:
    y_predict = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_predict)

import matplotlib as plt
plt.bar(name_models, roc_auc, color='blue')
plt.xticks(name_models)
plt.ylabel('ROC Curve')
plt.xlabel('Models')
plt.title('Results')
plt.show()