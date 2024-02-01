import pandas
import numpy

s = [
    [47, 94, 43, 92, 67, 19],
    [66, 52, 48, 79, 94, 44],
    [48, 21, 75, 14, 29, 56],
    [77, 10, 70, 42, 23, 62],
    [16, 10, 58, 93, 43, 53],
    [91, 60, 22, 46, 50, 41],
]
print(pandas.DataFrame(s))

s = {
    "a": [47, 66, 48, 77, 16, 91],
    "b": [94, 52, 21, 10, 10, 60],
    "c": [43, 48, 75, 70, 58, 22],
    "d": [92, 79, 14, 42, 93, 46],
    "e": [67, 94, 29, 23, 43, 50],
    "f": [19, 44, 56, 62, 55, 41],
}
data = pandas.DataFrame(s, columns=['a', 'b', 'c', 'd', 'e', 'f'])
print(data)
print("----------------返回a列--------------")
print(data[['a']])  # 返回a列，DataFrame格式
data.iloc[:, 0]  # 返回a列，Series格式
print(data.a)  # 返回a列，Series格式
data['a']  # 返回a列，Series格式
print("----------------pandas Dataframe行选择--------------")
print(data[1:2])
data.loc[1:1]
print(data.loc[1])  # 返回Series格式

data.iloc[-1:]
data[-1:]
print(data.tail(1))
print("----------------连续多行--------------")
print(data[2:5])
print(data.loc[2:4])
print("----------------不连续多行--------------")
print(data.iloc[[2, 3, 5], :])
print(data.iloc[[2, 3, 5], 1:4])
print("----------------连续前多行--------------")
print(data.head(2))
print("----------------连续末多行--------------")
print(data.tail(2))
print("----------------随机多行--------------")
print(data.sample(3))

print("--------------pandas Dataframe返回指定行列--------------")
print("--------------指定行列值--------------")
print(data.iat[1, 2])
print("--------------指定行列区域--------------")
print(data.iloc[[2, 3, 5], [0, 4]])

print("-------------pandas Dataframe条件查询--------------")
print("-------------单条件------------------")
print(data[data.a > 50])
print(data[data['a'] > 50])
data.loc[data.a > 50, :]
data.loc[data['a'] > 50, :]
print("-------------多条件------------------")
print(data.loc[(data.a > 40) & (data.b > 60), :])
print(data[(data.a > 40) & (data.b > 40)])
print("-------------单条件，多列------------------")
print(data.loc[data.a > 50, ['a', 'b', 'd']])
print(data.loc[data['a'] > 50, ['a', 'b', 'd']])

print("-------------pandas Dataframe聚合--------")
print("-------------各列的合--------")
print(data.sum(axis=1))
print("-------------平均数--------")
print(numpy.mean(data.values))
print("-------------各行的合--------")
print(data.sum(axis=0))
print("-------------聚合统计--------")
print(data.describe())
print("-------------分位数--------")
print(data.quantile(axis=0))  # 按列计算
print("-------------方差--------")
print(data.var(axis=1))  # 按行计算
print("-------------pandas Dataframe分组统计--------")

ss = [
    [47, 66, 48, 77, 16, 91, 'GD', 'GZ'],
    [94, 52, 21, 10, 10, 60, 'GD', 'SZ'],
    [43, 48, 75, 70, 58, 22, 'FJ', 'FZ'],
    [92, 79, 14, 42, 93, 46, 'FJ', 'FZ'],
    [67, 94, 29, 23, 43, 50, 'GX', 'NN'],
    [19, 44, 56, 62, 55, 41, 'GX', 'NN']
]
# 列名不能有空格，必须顶格
df = pandas.read_csv("groupbyTest.csv")
# df = pandas.DataFrame(ss, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
print(df)
# print(df.groupby('g').sum())
# 与上下方法调用互斥 不能同时使用
# print(df.groupby('g')['d'].agg([numpy.sum, numpy.mean, numpy.std]))
# print(df.groupby('g').agg([numpy.sum, numpy.mean, numpy.std]))
# print(df.groupby(['g', 'h']).mean())
print("-------------透视表分组统计--------")
print(pandas.pivot_table(df, index='g', values='a', columns=['h'], aggfunc="sum", fill_value=0, margins=True))

print("-------------pandas Dataframe处理缺失值--------")
s = {
    "a": [47, 66, 48, 77, 16, 91],
    "b": [94, 52, None, 10, 10, 60],
    "c": [43, 48, 75, 70, 58, 22],
    "d": [92, 79, 14, 42, 93, 46],
    "e": [67, 94, 29, 23, 43, None],
    "f": [19, 44, 56, 62, 55, 41],
}
data = pandas.DataFrame(s, columns=['a', 'b', 'c', 'd', 'e', 'f'])
print(data)
print(data.dropna(axis=0))
print(data.dropna(axis=1))
print(data.fillna(999))

s1 = {
    "id": [1, 2, 3, 4, 5, 6],
    "b": [12, 52, None, 10, 10, 60],
    "c": [32, 48, 75, 70, 58, 22]
}
df1 = pandas.DataFrame(s1, columns=['id', 'b', 'c'])
print(df1)
s2 = {
    "id": [1, 2, 3, 4, 5, 6],
    "e": [5, 94, 29, 23, 43, None],
    "f": [6, 44, 56, 62, 55, 41]
}
df2 = pandas.DataFrame(s2, columns=['id', 'e', 'f'])
print(df2)
print(pandas.merge(df1, df2, how='inner'))
print(pandas.merge(df1, df2, how='inner', left_index=True, right_index=True))