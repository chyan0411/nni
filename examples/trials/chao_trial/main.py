# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import nni
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import logging
import numpy as np

import multiprocessing
multiprocessing.set_start_method('forkserver')
import itertools
import pandas as pd
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics

dimport pandas as pd
import nni
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.model_selection import KFold
import pickle

# get_ipython().run_line_magic('matplotlib', 'inline')

    


FILE_PATH = 'Verdika_remove_unnecessary_done_v2.csv'
data = pd.read_csv(FILE_PATH)

X = data.drop(['Swapped'], axis = 1)
y = data.Swapped

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=99, test_size=0.2) 

# x_train = pickle.load(open("./data/x_train", 'rb'))
# x_test = pickle.load(open("./data/x_test", 'rb'))

# test = pd.read_csv("./data/test.csv")
# train = pd.read_csv("./data/train.csv")

# passengerId = test['PassengerId']
# y_train = train['Survived'].ravel()

# 获取默认参数
def get_default_parameters():
     params = {
          'learning_rate': 0.02,
          'n_estimators': 2000,
          'max_depth': 4,
          'min_child_weight':2,
          'gamma':0.9,
          'subsample':0.8,
          'colsample_bytree':0.8,
          'objective':'binary:logistic',
          'nthread':-1,
          'scale_pos_weight':1
     }
     return params

# 获取模型
def get_model(PARAMS):
     model = xgb.XGBClassifier()
     model.learning_rate = PARAMS.get("learning_rate")
     model.max_depth = PARAMS.get("max_depth")
     model.subsample = PARAMS.get("subsample")
     model.colsample_btree = PARAMS.get("colsample_btree")
     return model

# 运行模型
kf = KFold(n_splits=5)
def run(x_train, y_train, model):
     scores = cross_val_score(model, x_train, y_train, cv=kf)
     score = scores.mean()
     nni.report_final_result(score)

if __name__ == '__main__':
     RECEIVED_PARAMS = nni.get_next_parameter()
     PARAMS = get_default_parameters()
     PARAMS.update(RECEIVED_PARAMS)
     model = get_model(PARAMS)
     run(x_train, y_train, model)