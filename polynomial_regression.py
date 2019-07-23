# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:34:05 2019

@author: Caner.Tasan
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Caner.Tasan\\Desktop\\Datas\\polynomial_regression.csv")

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")


#linear regression  => y = b0 + b1*z 
#multiple linear regression => y = b0 + b1*x + b2*x

#%%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

#%%
y_head = lr.predict(x)

plt.plot(x, y_head, color = "red" , label = "linear")
#plt.show()

lr.predict([[10000]])

#%%
#polynomial linear regression  => y = b0 + b1*z + b2*x^2 + b3*x^3 + ... + bn*x^n

from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree=4)

x_polynomial = polynomial_regression.fit_transform(x) # uygula ve çevir.

#%% fit
linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)

#%% plot

y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x,y_head2,color = "green", label = "polynomial")
plt.legend()
plt.show()



