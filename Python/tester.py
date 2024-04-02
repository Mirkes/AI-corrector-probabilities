# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:09:28 2024

@author: em322
"""

import numpy as np
from guarantee import *
from random import sample
from math import floor

# Load data and prepare datasets
with open('QSAR.csv') as file:
    content = file.readlines()
# Remove headers
content = content[1:]
# Create tables for reading
data = np.zeros((len(content), 41))
outcome = np.zeros(len(content))
for k in range(len(content)):
    tmp = content[k].split(",")
    ttt = [float(i) for i in tmp]
    data[k, :] = np.asarray(ttt[0:-1])
    outcome[k] = ttt[-1]

# Test of oneDClass
# Index of positive (bad) outcome
ind = outcome == 1;
x = data[ind, :]
y = data[~ind, :]
accs = ["BA", "Acc", "f1", "NPV", "PPV", "TPR", "TNR"]
for ac in accs:
    [bestT, bestErr, dir] = fisher(x, y, ["LFD score for Binary GOSE outcome", "Bad", "Good"], ac)

[bestT, bestErr1, dir] = fisher(x, y, acc='ba')

#Test of randomly selected. It is infinite loop to find good direction. 
# Please decomment following rows to test, if you want
# bestErr = float('inf')
# bestDir = None
# siz = x.shape[1]
# while True:
#     dirs = np.random.rand(siz)
#     [t, err, dirs] = specDir(x, y, dirs, acc='BA')
#     if err < bestErr:
#         bestDir = dirs
#         bestErr = err
#         print(err)

# Split data into test and training set
ind = sample(range(x.shape[0]), floor(0.2 * x.shape[0]))
xTe = x[ind, :];
xTr = np.delete(x, ind, 0)
ind = sample(range(y.shape[0]), floor(0.2 * y.shape[0]))
yTe = y[ind, :];
yTr = np.delete(y, ind, 0)

# Form model for training set
[bestT, bestErr, direct] = fisher(xTr, yTr, None, 'ba');

# Assess this model for test set
xSc = np.matmul(xTe, direct);
ySc = np.matmul(yTe, direct)

# Form labels and predictions
labX = np.zeros(xSc.shape[0])
labY = np.ones(ySc.shape[0])
predX = np.zeros(xSc.shape[0])
predY = np.ones(ySc.shape[0])

# Form sets for estimation
ind = xSc >= bestT;
predX[ind] = 1
ind = ySc < bestT;
predY[ind] = 0

data = np.concatenate((xTe, yTe))
lab = np.concatenate((labX, labY))
pred = np.concatenate((predX, predY))
# According to documentations classes must be numerated from 1.
lab = lab+1;
pred = pred+1;

# Test of rejectAcceptModel creation
mdl = RejectAcceptModel(data, lab, pred) 
