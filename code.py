# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:16:36 2022

@author: tahmi
"""

import numpy as np
import pandas as pd
import missingno as msno
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import svm

# Loading Data
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slop', 'ca', 'thal', 'heartdisease']
clivelandData = pd.read_csv('Dataset/cleveland.csv', names = features)
hungarianData = pd.read_csv('Dataset/hungarian.csv', names = features)
switzerlandData = pd.read_csv('Dataset/switzerland.csv', names = features)

datatemp = [clivelandData, hungarianData, switzerlandData]
data = pd.concat(datatemp)

# Preprocessing Data
data = data.drop(['slop', 'ca', 'thal'], axis=1)
data = data.replace('?', np.nan)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imputedData = imp.fit_transform(data)