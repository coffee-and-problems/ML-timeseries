import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.arange(0,20).reshape((-1,1))
y = np.arange(0,20) + np.random.normal(0, 239, 20)

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

plt.scatter(x, y,  color='red')
plt.plot(x, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
