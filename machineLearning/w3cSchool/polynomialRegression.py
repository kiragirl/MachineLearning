import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 多项式回归（Polynomial Regression） https://www.w3school.com.cn/python/python_ml_polynomial_regression.asp
# 如果您的数据点显然不适合线性回归（穿过数据点之间的直线），那么多项式回归可能是理想的选择。
#
# 像线性回归一样，多项式回归使用变量 x 和 y 之间的关系来找到绘制数据点线的最佳方法
x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()
# 拟合度
print(r2_score(y, mymodel(x)))
speed = mymodel(17)
