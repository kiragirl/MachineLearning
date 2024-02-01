import pandas
from sklearn import linear_model

df = pandas.read_csv("cars.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

# 预测重量为 2300kg、排量为 1300ccm 的汽车的二氧化碳排放量：

predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2)
# 系数是描述与未知变量的关系的因子。
print(regr.coef_)
