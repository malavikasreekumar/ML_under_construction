# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 18:06:22 2019

@author: Malavika  Sreekumar
"""

# Upper Confidence Bound Algorithm in Reinforcement learning

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing datasets
datasets=pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
