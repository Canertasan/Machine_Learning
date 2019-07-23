# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:18:23 2019

@author: Caner.Tasan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\Caner.Tasan\\Desktop\\Datas\\column_2C_weka.csv")

data1 = data[data['class'] =='Abnormal']

x = np.array(data1.loc[: , 'pelvic_incidence']).reshape(-1,1)
y = np.array(data1.loc[:, 'sacral_slope']).reshape(-1,1)

plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show();

#%% linear regression
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

predict_space = np.linspace(min(x),max(x))

linear_reg.fit(x,y)

predicted = linear_reg.predict(predict_space)

#%% score

print('R^2 score: ',linear_reg.score(x, y))

# Plot regression line and scatter
plt.plot(predict_space, predicted, color='black', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()



