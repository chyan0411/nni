


import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
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

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import nni

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support

FILE_PATH = 'Verdika_remove_unnecessary_done_v2.csv'
# FILE_PATH = 'C:\\Users\\cyan\\Documents\\nni\\examples\\trials\\chao_trial\\Verdika_remove_unnecessary_done_v2.csv'
data = pd.read_csv(FILE_PATH)


X = data.drop(['Swapped'], axis = 1)
y = data.Swapped

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 

# scoring = {'accuracy' : make_scorer(accuracy_score), 
#            'precision' : make_scorer(precision_score),
#            'recall' : make_scorer(recall_score), 
#            'f1_score' : make_scorer(f1_score)}

def get_default_parameters():
     params = {
          'learning_rate': 0.02,
          'n_estimators': 20,
          'max_depth': 4
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
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
def run(x_train, y_train, model):
     scores = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1, error_score='raise')
     # score = scores.mean()
     score = np.mean(scores)
     # print(score)
     nni.report_final_result(score)
     # nni.report_intermeidate_result(score)
     # y_preds = model.predict(x_test)

     # print('\nClassification Report: ')
     # report = classification_report(y_test, y_preds)

     # nni.report_final_result(report)


if __name__ == '__main__':
     RECEIVED_PARAMS = nni.get_next_parameter()
     PARAMS = get_default_parameters()
     PARAMS.update(RECEIVED_PARAMS)
     model = get_model(PARAMS)
     run(X_train, y_train, model)