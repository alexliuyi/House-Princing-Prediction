# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:39:55 2019

@author: alexliuyi
"""

import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model  import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error

import config


def clean_data(raw_data):
    """
        Raw Input:
            - raw_data: Raw Data

        Return:
            - cln_data: Cleaned Data
    """ 
    cln_data = raw_data.dropna(how='any', axis=0)
    
    return cln_data


def inspect_dataset(train_data, test_data):
    """
        Check Data
    """
    print('\n===================== Checking =====================')
    print('Observations in Training Data: {}'.format(len(train_data)))
    print('Observations in Testing Data:  {}'.format(len(test_data)))


def transform_train_data(train_data):
    """
        Feature Engineering
        1. OneHot 
        2. Scaler

        Paras:
            - train_data:

        Returns:
            - X_train:    
            - label_encs:  
            - onehot_enc: 
            - scaler:     
            - pca:       
    """
    label_encs = []
    onehot_enc = OneHotEncoder(sparse=False, categories='auto')
    scaler = MinMaxScaler()

    # Categorical Variables
    label_feats = None
    for cat_col in config.cat_cols:
        label_enc = LabelEncoder()
        label_feat = label_enc.fit_transform(train_data[cat_col].values).reshape(-1, 1)
        if label_feats is None:
            label_feats = label_feat
        else:
            label_feats = np.hstack((label_feats, label_feat))
        label_encs.append(label_enc)

    onehot_feats = onehot_enc.fit_transform(label_feats)

    # Numerical Variables
    numeric_feats = train_data[config.numeric_cols].values

    # Combine
    all_feats = np.hstack((onehot_feats, numeric_feats))

    # Scale
    scaled_all_feats = scaler.fit_transform(all_feats)

    print('Dimension after transform: {}(Categorical: {}，Numerical: {}）'.format(
        scaled_all_feats.shape[1], onehot_feats.shape[1], numeric_feats.shape[1]))

    # Dimension Reduction
    pca = PCA(n_components=0.99)
    X_train = pca.fit_transform(scaled_all_feats)

    print('Dimension after PCA: {}'.format(X_train.shape[1]))

    return X_train, label_encs, onehot_enc, scaler, pca


def transform_test_data(test_data, label_encs, onehot_enc, scaler, pca):
    """
        Paras:
            - test_data: 
            - label_encs: 
            - onthot_enc:
            - scaler:    
            - pca:       

        Returns:
            - X_Test:    
    """
    # Categorical Variables
    label_feats = None
    for i, cat_col in enumerate(config.cat_cols):
        label_enc = label_encs[i]
        label_feat = label_enc.transform(test_data[cat_col].values).reshape(-1, 1)
        if label_feats is None:
            label_feats = label_feat
        else:
            label_feats = np.hstack((label_feats, label_feat))

    onehot_feats = onehot_enc.transform(label_feats)

    # Numerical Variables
    numeric_feats = test_data[config.numeric_cols].values

    # Combine
    all_feats = np.hstack((onehot_feats, numeric_feats))

    # Scale
    scaled_all_feats = scaler.transform(all_feats)

    # Dimension Reduction
    X_test = pca.transform(scaled_all_feats)

    return X_test


def train_test_model(X_train, y_train, X_test, y_test, model_name, model):
    """
        model_name:
            LR: Linear regression
            Lasso:
            Ridge:

    """
    print('Training {}...'.format(model_name))
    clf = model
    start = time.time()
    clf.fit(X_train, y_train)
    # timing
    end = time.time()
    duration = end - start
    print('Timing {:.4f}s'.format(duration))

    # Metric
    y_train_pred = clf.predict(X_train)
    train_score = mean_squared_log_error(y_train, y_train_pred)
    print('Training log MSE: {:.8f}'.format(train_score))

    y_test_pred = clf.predict(X_test)
    test_score = mean_squared_log_error(y_test, y_test_pred)
    print('Timing{:.4f}s'.format(test_score))
    print('Testing log MSE: {:.8f}'.format(duration))
    print()

    return clf, test_score, duration


