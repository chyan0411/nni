import nni
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import logging
import numpy as np
import pandas as pd

LOG = logging.getLogger('sklearn_classification')

def load_data():
    FILE_PATH = 'Verdika_remove_unnecessary_done_v2.csv'
    data = pd.read_csv(FILE_PATH)

    X = data.drop(['Swapped'], axis = 1)
    y = data.Swapped
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=99, test_size=0.25)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    return X_train, X_test, y_train, y_test

def get_default_parameters():
    '''get default parameters'''
    params = {
        'C': 1.0,
        'keral': 'linear',
        'degree': 3,
        'gamma': 0.01,
        'coef0': 0.01
    }
    return params

def get_model(PARAMS):
    '''Get model according to parameters'''
    model = SVC()
    model.C = PARAMS.get('C')
    model.keral = PARAMS.get('keral')
    model.degree = PARAMS.get('degree')
    model.gamma = PARAMS.get('gamma')
    model.coef0 = PARAMS.get('coef0')
    
    return model

def run(X_train, X_test, y_train, y_test, PARAMS):
    '''Train model and predict result'''
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    LOG.debug('score: %s' % score)
    nni.report_final_result(score)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS)
        run(X_train, X_test, y_train, y_test, model)
    except Exception as exception:
        LOG.exception(exception)
        raise