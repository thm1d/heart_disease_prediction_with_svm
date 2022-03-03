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
from sklearn.metrics import accuracy_score

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

# Creating Test and Train Data
X_train, X_test, y_train, y_test = train_test_split(imputedData[:, :-1], imputedData[:, -1], test_size=0.3, random_state=42)


# Scale Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and Train the Model
classifier = svm.SVC(kernel='rbf')
classifier.fit(X_train, y_train)
preds = classifier.predict(X_test)

# Accuracy of Predictions
score = accuracy_score(y_test, preds)
print("Prediction Accuracy is : %f" %(score * 100))