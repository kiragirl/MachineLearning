import matplotlib
import matplotlib.pyplot as plt
from pasta.augment import inline
import seaborn as sns
import numpy as np
'''
x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
y_bar = [3, 4, 6, 8, 9, 10, 9, 11, 7, 8]
y_line = [2, 3, 5, 7, 8, 9, 8, 10, 6, 7]
plt.bar(x, y_bar)
plt.plot(x, y_line, '-o', color='y')
plt.show()


sns.set()  # 声明使用 Seaborn 样式
plt.bar(x, y_bar)
plt.plot(x, y_line, '-o', color='y')
plt.show()
'''
iris = sns.load_dataset("iris")
print(iris.head())
'''
print("--------------------散点图-----------------------")
sns.relplot(x="sepal_length", y="sepal_width", data=iris)
sns.relplot(x="sepal_length", y="sepal_width", hue="species", data=iris)
sns.relplot(x="sepal_length", y="sepal_width", hue="species", style="species", data=iris)
sns.relplot(x="petal_length", y="petal_width", hue="species", style="species", data=iris)
print("--------------------线行图-----------------------")
sns.relplot(x="sepal_length", y="petal_length", hue="species", style="species", kind="line", data=iris)
sns.lineplot(x="sepal_length", y="petal_length", hue="species", style="species", data=iris)

print("--------------------类别图-----------------------")
sns.catplot(x="sepal_length", y="species", hue="species", data=iris)

sns.catplot(x="sepal_length", y="species",  hue="species", kind="swarm", data=iris)

print("--------------------箱线图 box-----------------------")
sns.catplot(x="sepal_length", y="species", kind="box", data=iris)

print("--------------------增强箱线图 boxen-----------------------")
sns.catplot(x="species", y="sepal_length", hue="species", kind="boxen", data=iris)

print("--------------------小提琴图 violin-----------------------")
sns.catplot(x="sepal_length", y="species", kind="violin", data=iris)

print("--------------------分布图-----------------------")
sns.distplot(iris["sepal_length"])
sns.distplot(iris["sepal_length"], kde=False)

print("--------------------二元变量分布图 jointplot-----------------------")
sns.jointplot(x="sepal_length", y="sepal_width", data=iris)
print("--------------------绘制出核密度估计对比图 kde-----------------------")
sns.jointplot(x="sepal_length", y="sepal_width", hue="species", data=iris, kind="kde")

print("--------------------绘制六边形计数图 hex-----------------------")
sns.jointplot(x="sepal_length", y="sepal_width", data=iris, kind="hex")

print("--------------------绘制回归拟合图 reg-----------------------")

sns.jointplot(x="sepal_length", y="sepal_width",data=iris, kind="reg")

print("--------------------变量两两对比图 pairplot----------------------")
sns.pairplot(iris, hue="species")

print("--------------------回归图----------------------")
sns.lmplot(x="sepal_length", y="sepal_width", hue="species", data=iris)
plt.show()
'''
print("--------------------矩阵图----------------------")
sns.heatmap(np.random.rand(10, 10))
plt.show()
iris.pop("species")
sns.clustermap(iris)