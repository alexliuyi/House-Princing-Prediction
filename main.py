# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:21:46 2019

@author: alexliuyi
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model  import LinearRegression, Lasso, Ridge, SGDRegressor, ARDRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn import tree
from sklearn import ensemble
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

import config
import utils


def main():
    # Loading Data
    cleaned = pd.read_csv(os.path.join(config.dataset_path, 'cleaned.csv'))
    cleaned = cleaned[config.feat_cols]
    
    raw_data = cleaned[pd.isna(cleaned.y)==False]
    valid = cleaned[pd.isna(cleaned.y)]

    # Cleaning Data

    # Splitting Data
    train_data, test_data = train_test_split(raw_data, test_size=1/4, random_state=10)
    y_train = train_data['y'].values
    y_test = test_data['y'].values

    train_data = train_data.drop('y', axis=1)
    test_data  = test_data.drop('y', axis=1)
    
    

    # Checking Data
    utils.inspect_dataset(train_data, test_data)

    # Feature Engineering
    print('\n===================== Feature Engineering =====================')
    X_train, label_encs, onehot_enc, scaler, pca = utils.transform_train_data(train_data) 
    X_test = utils.transform_test_data(test_data, label_encs, onehot_enc, scaler, pca)
    X_valid = utils.transform_test_data(valid, label_encs, onehot_enc, scaler, pca)
    
    # 构建训练测试数据
    # 数据建模及验证
    print('\n===================== Modeling =====================')
    model_name_param_dict = {'LR':    LinearRegression(),
                             'Lasso': Lasso(alpha=0.01),
                             'Ridge': Ridge(alpha=0.01),
                             'SVM':SVR(),
                             'SGD':SGDRegressor()
                             }

    # 比较结果的DataFrame
    results_df = pd.DataFrame(columns=['MSE', 'Time (s)'],
                              index=list(model_name_param_dict.keys()))
    results_df.index.name = 'Model'
    for model_name, model in model_name_param_dict.items():
        _, best_score, mean_duration = utils.train_test_model(X_train, y_train, X_test, y_test, model_name, model)
        results_df.loc[model_name, 'MSE'] = best_score
        results_df.loc[model_name, 'Time (s)'] = mean_duration

    results_df.to_csv(os.path.join(config.output_path, 'table.csv'))
    
    lasso = Ridge(alpha=0.01)
    lasso.fit(X_train, y_train)
    predict = np.exp(lasso.predict(X_valid))
    result = pd.DataFrame(predict)
    result.to_csv('C:\\Users\\alexliuyi\\Documents\\Kaggle\\Home Price\\result.csv')

    # Validation Data

if __name__ == '__main__':
    main()
