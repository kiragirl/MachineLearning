import numpy
from scipy import stats
import matplotlib.pyplot as plt

speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
# 平均数
print(numpy.mean(speed))
# 中值 中值是对所有值进行排序后的中间值
print(numpy.median(speed))
# 众数 众值是出现次数最多的值
print(stats.mode(speed))
# 查找标准差
print(numpy.std(speed))
# 方差
print(numpy.var(speed))
# 百分位数
ages = [5, 31, 43, 48, 50, 41, 7, 11, 15, 39, 80, 82, 32, 2, 8, 6, 25, 36, 27, 61, 31]
print(numpy.percentile(ages, 75))
print(numpy.percentile(ages, 90))

# 绘制直方图
# plt.hist(numpy.random.uniform(0.0, 5.0, 100000), 100)
# plt.show()

# 直方图解释 正太数据分布
# 我们使用 numpy.random.normal() 方法创建的数组（具有 100000 个值）绘制具有 100 栏的直方图。

# 我们指定平均值为 5.0，标准差为 1.0。

# 这意味着这些值应集中在 5.0 左右，并且很少与平均值偏离 1.0。

# 从直方图中可以看到，大多数值都在 4.0 到 6.0 之间，最高值大约是 5.0。
plt.hist(numpy.random.normal(5.0, 1.0, 100000), 100)
plt.show()

# 散点图
# x 数组代表每辆汽车的年龄。
# y 数组表示每个汽车的速度。
# x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
# y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

# plt.scatter(x, y)
# plt.show()

# x = numpy.random.normal(5.0, 1.0, 1000)
# y = numpy.random.normal(10.0, 2.0, 1000)

# plt.scatter(x, y)
# plt.show()

# 线性回归 https://www.w3school.com.cn/python/python_ml_linear_regression.asp

x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

slope, intercept, r, p, std_err = stats.linregress(x, y)
# 相关性
print(slope)
print(intercept)
print(r)
print(p)


def myfunc(x):
    return slope * x + intercept


mymodel = list(map(myfunc, x))

# 绘制图像
# plt.scatter(x, y)
# plt.plot(x, mymodel)
# plt.show()
# 预测
speed = myfunc(10)
print(speed)

