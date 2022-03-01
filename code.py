# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:33:54 2022

@author: tahmid
"""

import numpy as np
import pandas as pd
import missingno as msno
import seaborn as sn
import matplotlib.pyplot as plt

# Loading Data
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slop', 'ca', 'thal', 'heartdisease']
clivelandData = pd.read_csv('Dataset/cleveland.csv', names = features)
hungarianData = pd.read_csv('Dataset/hungarian.csv', names = features)
switzerlandData = pd.read_csv('Dataset/switzerland.csv', names = features)

datatemp = [clivelandData, hungarianData, switzerlandData]
data = pd.concat(datatemp)