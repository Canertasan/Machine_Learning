# -*- coding: utf-8 -*-

#import library
import pandas as pd
import matplotlib.pyplot as plt

#import data
df = pd.read_csv("C:\\Users\\Caner.Tasan\\Desktop\\Datas\\linear_regression_dataset.csv")

#plot data
plt.scatter(df.Deneyim, df.maas)
plt.xlabel("Deneyim")
plt.ylabel("maas")
plt.show()

#%%  linear_regression

# sklearn library
from sklearn.linear_model import LinearRegression

# linear_regression_model
linear_reg = LinearRegression()

x = df.Deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

#%% prediction
import numpy as np
b0 = linear_reg.predict([[0]])

print("b0: ",b0)

b0_ = linear_reg.intercept_
print("b0_: ",b0_)  #intercept

b1 = linear_reg.coef_
print("b1: ",b1)  #slope

# maas = 1663 + 1138*deneyim

print(linear_reg.predict([[11]]))


array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1) # deneyim

plt.scatter(x,y)
#plt.show()

y_head = linear_reg.predict(array) #maas

plt.plot(array, y_head, color = "red")

linear_reg.predict([[100]])

