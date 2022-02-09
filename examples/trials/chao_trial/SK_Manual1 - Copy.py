#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing
# multiprocessing.set_start_method('forkserver')
import itertools
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
# import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics 

from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

# get_ipython().run_line_magic('matplotlib', 'inline')

# import nni


FILE_PATH = 'C:\\Users\\cyan\\Documents\\nni\\examples\\trials\\chao_trial\\Verdika_remove_unnecessary_done_v2.csv'
data = pd.read_csv(FILE_PATH)


X = data.drop(['Swapped'], axis = 1)
y = data.Swapped

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 

# model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=0, verbose = 1) 

def get_default_parameters():
     params = {
          'learning_rate': 0.02,
          'n_estimators': 20,
          'max_depth': 2
     }
     return params

def get_model(PARAMS):
    model = GradientBoostingClassifier()
    model.n_estimators = PARAMS.get("n_estimators")
    model.learning_rate = PARAMS.get("learning_rate")
    model.max_depth = PARAMS.get("max_depth")
    return model

# model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=0, verbose = 1) 

# 运行模型
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)
def run(x_train, y_train, model):
     scores = cross_val_score(model, X_train, y_train, scoring='roc_auc_ovr_weighted', cv=cv, n_jobs=-1, error_score='raise')
     score = scores.mean()
     print(score)
     # nni.report_final_result(score)



if __name__ == '__main__':
     # RECEIVED_PARAMS = nni.get_next_parameter()
     PARAMS = get_default_parameters()
     # PARAMS.update(RECEIVED_PARAMS)
     model = get_model(PARAMS)
     run(X_train, y_train, model)