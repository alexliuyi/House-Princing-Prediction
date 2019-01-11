# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:02:13 2019

@author: alexliuyi
"""

import os

os.chdir("C:\\Users\\alexliuyi\\Documents\\Kaggle\\Home Price")

# Data Path

dataset_path = 'C:\\Users\\alexliuyi\\Documents\\Kaggle\\Home Price\\data\\'

# Output Path
output_path = 'C:\\Users\\alexliuyi\\Documents\\Kaggle\\Home Price\\output\\'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Numberic Variables
numeric_cols = ['MSSubClass', 'OverallQual', 'OverallCond', 'BsmtFinSF1',
               'BsmtFullBath', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr',
               'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'lotarea_new',
               'grlivarea_new', '_1stflrsf_new', 'age_built', 'age_remode']

# Categorical Variables
cat_cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'LotConfig',
           'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle',
           'Exterior1st', 'Exterior2nd', 'ExterQual', 'Foundation', 'BsmtQual',
           'BsmtCond', 'BsmtExposure', 'HeatingQC', 'CentralAir', 'Electrical',
           'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageQual', 'GarageCond',
           'PavedDrive', 'Fence', 'SaleType', 'SaleCondition']

y = ['y']

feat_cols = numeric_cols + cat_cols + y




