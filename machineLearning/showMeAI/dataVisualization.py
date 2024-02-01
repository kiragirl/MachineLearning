import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
df = pd.DataFrame(np.random.randn(10, 4), index=pd.date_range('1/1/2022',
                                                              periods=10), columns=list('ABCD'))

print("-------------line chart------------")
df.plot()
plt.show()

print("-------------Column chart------------")
df = pd.DataFrame(np.random.rand(10, 4), columns=['e', 'f', 'g', 'h'])
df.plot.bar()
plt.show()

df.plot.bar(stacked=True)
plt.show()

print("-------------bar chart------------")
df = pd.DataFrame(np.random.rand(10, 4), columns=['e', 'f', 'g', 'h'])
df.plot.barh(stacked=True)
plt.show()
print("-------------histogram 直方图------------")
df = pd.DataFrame({'a':np.random.randn(1000)+1,'b':np.random.randn(1000),'c':np.random.randn(1000) - 1, 'd':np.random.randn(1000) -2}, columns=['a', 'b', 'c', 'd'])
df.plot.hist(bins=20)
plt.show()

print("-------------box plot 箱形图------------")
df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
df.plot.box()
plt.show()


print("-------------area char 面积图------------")
# 数字作为横轴坐标时 失效
df = pd.DataFrame(np.random.rand(5, 4), index=['Mon', 'Tue', 'Wen', 'Thur', 'Fri'], columns=['A', 'B', 'C', 'D'])
df.plot.area()
plt.show()

print("-------------scatter Chart 散点图-----------")
df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
df.plot.scatter(x='a', y='b')
plt.show()
'''
print("-------------pie Chart 饼状图-----------")
df = pd.DataFrame(3 * np.random.rand(4), index=['a', 'b', 'c', 'd'], columns=['x'])
df.plot.pie(subplots=True)
plt.show()