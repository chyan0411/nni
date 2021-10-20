#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing
multiprocessing.set_start_method('forkserver')
import itertools
import numpy as np
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

get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


FILE_PATH = 'Verdika_remove_unnecessary_done_v2.csv'
data = pd.read_csv(FILE_PATH)

data.head(5)

X = data.drop(['Swapped'], axis = 1)
y = data.Swapped

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=0, verbose = 1) 



cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring='roc_auc_ovr_weighted', cv=cv, n_jobs=-1, error_score='raise')
print('AUC: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support
y_preds = model.predict(X_test)
print('Precision                                   : %.3f'%precision_score(y_test, y_preds))
print('Recall                                      : %.3f'%recall_score(y_test, y_preds))
print('F1-Score                                    : %.3f'%f1_score(y_test, y_preds))
print('\nPrecision Recall F1-Score Support Per Class : \n',precision_recall_fscore_support(y_test, y_preds))
print('\nClassification Report                       : ')
print(classification_report(y_test, y_preds))

metrics.roc_auc_score(y_test, y_preds, average='weighted')


# In[ ]:





# In[ ]:





# In[3]:


FILE_PATH = 'Verdika_remove_unnecessary_done_v2.csv'
data = pd.read_csv(FILE_PATH)

data.head(5)

X = data.drop(['Swapped'], axis = 1)
y = data.Swapped

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 

model = RandomForestClassifier(n_estimators=10, 
                               max_depth=5, 
                               random_state=0, 
                               verbose = 1, 
                               min_samples_leaf= 10
                              )

# random forest for classification in scikit-learn


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring='roc_auc_ovr_weighted', cv=cv, n_jobs=-1, error_score='raise')
print('AUC: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support
y_preds = model.predict(X_test)
print('Precision                                   : %.3f'%precision_score(y_test, y_preds))
print('Recall                                      : %.3f'%recall_score(y_test, y_preds))
print('F1-Score                                    : %.3f'%f1_score(y_test, y_preds))
print('\nPrecision Recall F1-Score Support Per Class : \n',precision_recall_fscore_support(y_test, y_preds))
print('\nClassification Report                       : ')
print(classification_report(y_test, y_preds))

metrics.roc_auc_score(y_test, y_preds, average='weighted')


# In[4]:


from sklearn.metrics import balanced_accuracy_score 
print('Balanced Accuracy          : ',balanced_accuracy_score(y_test, y_preds))


# In[ ]:





# In[6]:


FILE_PATH = 'Verdika_remove_unnecessary_done_v2.csv'
data = pd.read_csv(FILE_PATH)

data.head(5)

X = data.drop(['Swapped'], axis = 1)
y = data.Swapped

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 

model = RandomForestClassifier(n_estimators=10, 
                               max_depth=5, 
                               random_state=0, 
                               verbose = 1, 
                               min_samples_leaf= 10
                              )

# random forest for classification in scikit-learn


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring='roc_auc_ovr_weighted', cv=cv, n_jobs=-1, error_score='raise')
print('AUC: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support
y_preds = model.predict(X_test)
print('Precision                                   : %.3f'%precision_score(y_test, y_preds))
print('Recall                                      : %.3f'%recall_score(y_test, y_preds))
print('F1-Score                                    : %.3f'%f1_score(y_test, y_preds))
print('\nPrecision Recall F1-Score Support Per Class : \n',precision_recall_fscore_support(y_test, y_preds))
print('\nClassification Report                       : ')
print(classification_report(y_test, y_preds))

metrics.roc_auc_score(y_test, y_preds, average='weighted')


# In[7]:



from xgboost import XGBClassifier 

FILE_PATH = 'Verdika_remove_unnecessary_done_v2.csv'
data = pd.read_csv(FILE_PATH)

data.head(5)

X = data.drop(['Swapped'], axis = 1)
y = data.Swapped

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 

model = XGBClassifier(n_estimators=10, 
                      max_depth=5, 
                      random_state=0, 
                      verbose = 1,
                      learning_rate = 0.01,
                      n_jobs=-1)

# random forest for classification in scikit-learn


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring='roc_auc_ovr_weighted', cv=cv,  error_score='raise')
print('AUC: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support
y_preds = model.predict(X_test)
print('Precision                                   : %.3f'%precision_score(y_test, y_preds))
print('Recall                                      : %.3f'%recall_score(y_test, y_preds))
print('F1-Score                                    : %.3f'%f1_score(y_test, y_preds))
print('\nPrecision Recall F1-Score Support Per Class : \n',precision_recall_fscore_support(y_test, y_preds))
print('\nClassification Report                       : ')
print(classification_report(y_test, y_preds))

metrics.roc_auc_score(y_test, y_preds, average='weighted')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




