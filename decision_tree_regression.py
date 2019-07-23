# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 08:45:48 2019

@author: Caner.Tasan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("C:\\Users\\Caner.Tasan\\Desktop\\Datas\\decision_tree_regression_dataset.csv", header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%% decision tree regression
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor() # random state = 0
tree_reg.fit(x,y)

#tree_reg.predict([[6]])
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = tree_reg.predict(x_)
#%% visuilize
plt.scatter(x,y,color = "red")
plt.plot(x_,y_head,color = "green")
plt.xlabel("tribün level")
plt.ylabel("ücret")
plt.show()

















