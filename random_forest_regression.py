# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:24:40 2019

@author: Caner.Tasan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("C:\\Users\\Caner.Tasan\\Desktop\\Datas\\decision_tree_regression_dataset.csv" , header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

# %% Random forest
from sklearn.ensemble import RandomForestRegressor
# n_estimators kaç tane tree kullancaksak , random_State n sayıda seçim yapmak için yazıyoruz.
# eğer 2 kere run edersek ve random state yazmazsa 2 sonuc farklı cıkar!
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(x,y)

y_head = rf.predict(x)

# %% r-square method

from sklearn.metrics import r2_score

print("r_score: " , r2_score(y,y_head))

"""
print("7.8 seviyesinde fiyatın ne kadar olduğu: " , rf.predict([[7.8]]))


x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)
"""
#%%
"""
plt.scatter(x,y,color = "red")
plt.plot(x_,y_head,color = "green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()
"""



