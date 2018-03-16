# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 12:21:56 2017

@author: kd
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
datasets = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(datasets.values[i, j]) for j in range(0, 20)])
    
# Training the Apriorion the datasets
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
    

# Visualising the results
results = list(rules)
