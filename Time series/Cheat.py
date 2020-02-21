import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

file = 'GlobalTemperatures.csv'
df = pd.read_csv(file, parse_dates = ['dt'], index_col = ['dt'], names=['dt', 'LandAverageTemperature'], header=0)
series = df.LandAverageTemperature


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(series, model='additive')
fig = decomposition.plot()
plt.show()