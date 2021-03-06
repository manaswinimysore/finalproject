#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:31:09 2019

@author: manaswini
"""

import numpy as np
from urllib.request import urlopen
from scipy import stats

import urllib
import matplotlib.pyplot as plt # Visuals
import seaborn as sns 
import sklearn as skl
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier # Kth Nearest Neighbor
from sklearn.tree import DecisionTreeClassifier # Decision Trees
from sklearn.tree import export_graphviz # Extract Decision Tree visual
from sklearn.tree import tree 
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn import metrics
from sklearn.metrics import roc_curve # ROC Curves
from sklearn.metrics import auc # AUC 
from sklearn.model_selection import KFold, cross_val_score #cross validation 
from sklearn import cross_validation  #cross validation 
from urllib.request import urlopen # Get data from UCI Machine Learning Repository
import plotly.graph_objs as go
import plotly.plotly as py
from sklearn.preprocessing import PolynomialFeatures

def normalization(disease):
    dfnorm=disease.copy()
    for i in disease.columns:
        maxv=disease[i].max();
        minv=disease[i].min();
        dfnorm[i]=(disease[i]-np.mean(disease[i]))/(maxv-minv);
    return dfnorm
dataset = pd.read_csv('/Users/manaswini/Desktop/household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])


'''starting of preprocessing process'''
dataset.replace('?',np.nan,inplace=True)
dataset=dataset.astype('float32')

'''filling these null values with previous day data'''
def fill_missing(values):
    one_day = 60 * 24
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if np.isnan(values[row, col]):
                values[row, col] = values[row - one_day, col]

fill_missing(dataset.values);
daily_groups = dataset.resample('D')
daily_data = daily_groups.sum()



daily_data.to_csv('/Users/manaswini/Desktop/household_power_consumption_days.csv')
dataset = pd.read_csv('/Users/manaswini/Desktop/household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
dfnorm=normalization(dataset);
df = pd.read_csv('/Users/manaswini/Desktop/household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

X_train = df.drop(['Global_active_power'],axis=1)
Y_train = df['Global_active_power']

model =LinearRegression()
model.fit(X_train, Y_train)


def predict(Global_reactive_power,Voltage,Global_intensity,Sub_metering_1,Sub_metering_2,Sub_metering_3):
    person = np.array([[Global_reactive_power,Voltage,Global_intensity,Sub_metering_1,Sub_metering_2,Sub_metering_3]])

    P_prediction = model.predict(person)
    return P_prediction;
