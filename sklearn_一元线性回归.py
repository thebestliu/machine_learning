from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 载入数据
data = np.genfromtxt("E:\\python-m1\\regression\\data.csv", delimiter=",")
x_data = data[:, 0]
y_data = data[:, 1]
plt.scatter(x_data, y_data)
plt.show()
print(x_data.shape)

x_data = data[:, 0, np.newaxis]
print(x_data.shape)

x_data = data[:, 0, np.newaxis]
y_data = data[:, 1, np.newaxis]
# 创建并拟合数据
model = LinearRegression()
model.fit(x_data, y_data)

# 画图
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(x_data), 'r')
plt.show()
