import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
#Посмотрим на данные

file = 'GlobalTemperatures.csv'
df = pd.read_csv(file, parse_dates = ['dt'],
         index_col = ['dt'], names=['dt', 'LandAverageTemperature'], header=0)
series = df.LandAverageTemperature
print(df.head())
plt.xlabel('Date')
plt.ylabel('Land Average Temperature')
plt.plot(df)

plt.show()

#Проверим, является ли ряд стационарным с помощью теста Дикки-Фуллера
#https://ru.wikipedia.org/wiki/Тест_Дики_—_Фуллера
#Нулевая гипотиза - нестационарность ряда

result = adfuller(series.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

##Убираем сезонность

series_no_seasonal = series
for i in range(0, 12):
    series_no_seasonal = series_no_seasonal.diff()

result = adfuller(series_no_seasonal.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

S = 12 
#Исходный ряд
fig, axes = plt.subplots(3, 2, sharex=False)
axes[0, 0].plot(series); axes[0, 0].set_title('Original Series')
plot_acf(series, axes[0, 1])

#Ряд стал стационарным. А что, если бы нет?
axes[1, 0].plot(series_no_seasonal); axes[1, 0].set_title('No seasonal')
plot_acf(series_no_seasonal.dropna(),axes[1, 1])

#Первая разность
axes[2, 0].plot(series_no_seasonal.diff()); axes[2, 0].set_title('1st Differencing')
plot_acf(series_no_seasonal.diff().dropna(), axes[2, 1])

plt.show()

d = 0
D = 1

fig, axes = plt.subplots(1, 2, sharex=False)
axes[0].plot(series_no_seasonal); axes[0].set_title('No seasonal')
axes[1].set(ylim=(0,5))
plot_pacf(series_no_seasonal.dropna(), axes[1])

plt.show()

p = 2 
P = 0

fig, axes = plt.subplots(1, 2, sharex=False)
axes[0].plot(series_no_seasonal); axes[0].set_title('No seasonal')
axes[1].set(ylim=(0,1.2))
plot_acf(series_no_seasonal.dropna(), ax=axes[1])

plt.show()

q = 1
Q = 0

from statsmodels.tsa.statespace.sarimax import SARIMAX

##Гиперпараметры (p,d,q)x(P,D,Q)m
## (2,0,1)x(0,1,0)12 SARIMA Model
model = SARIMAX(series, order=(p,d,q), seasonal_order=(P,D,Q,S))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# Actual vs Fitted
fig, ax = plt.subplots(figsize=(9,4))
npre = 4
ax.set(xlabel='Date', ylabel='Temp')

series.loc['1900-01-01':].plot(ax=ax, style='o', label='Observed')
predicted = model_fit.predict(0, len(series)+100)
predicted.loc['1900-01-01':].plot(ax=ax, style='-', label='Predicted')

plt.show()