# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:09:28 2024

@author: em322
"""

import numpy as np
from guarantee import *

# For cross debugging data was specially saved and now we will upload it

# Load data and prepare datasets
with open('data.csv') as file:
    content = file.readlines()
# No headers
# Create tables for reading
data = np.zeros((len(content), 41))
for k in range(len(content)):
    tmp = content[k].split(",")
    ttt = [float(i) for i in tmp]
    data[k, :] = np.asarray(ttt)
    
# Load labels and prepare datasets
with open('lab.csv') as file:
    content = file.readlines()
# No headers
# Create tables for reading
lab = np.zeros(len(content))
for k in range(len(content)):
    lab[k] = float(content[k])

# Load labels and prepare datasets
with open('pred.csv') as file:
    content = file.readlines()
# No headers
# Create tables for reading
pred = np.zeros(len(content))
for k in range(len(content)):
    pred[k] = float(content[k])

# Test of rejectAcceptModel creation
mdl = RejectAcceptModel(data, lab, pred)

# Test of general prediction
# gen = mdl.generalEstimate()
