import numpy as np
import pandas as pd

boolean = [True, False]
gender = ["男", "女"]
color = ["white", "black", "yellow"]
data = pd.DataFrame({
    "height": np.random.randint(150, 190, 100),
    "weight": np.random.randint(40, 90, 100),
    "smoker": [boolean[x] for x in np.random.randint(0, 2, 100)],
    "gender": [gender[x] for x in np.random.randint(0, 2, 100)],
    "age": np.random.randint(15, 90, 100),
    "color": [color[x] for x in np.random.randint(0, len(color), 100)]
}
)
print(data)

print("-------------使用字典映射的map原理-------------")
data["gender"] = data["gender"].map({"男": 1, "女": 0})
print(data)
print("-------------使用函数映射的map原理-------------")


def gender_map(x):
    gender = '男' if x == 0 else '女'
    return gender


# 注意这里传入的是函数名，不带括号
data["gender"] = data["gender"].map(gender_map)
print(data)
print("-------------使用apply-------------")


def apply_age(x, bias):
    return x + bias


# 以元组的方式传入额外的参数
data["age"] = data["age"].apply(apply_age, args=(-3,))
print(data)
print("-------------DataFrame数据处理-------------")

print("-------------沿着0轴求和-------------")
print(data[["height", "weight", "age"]].apply(np.sum, axis=0))
print("-------------沿着0轴取对数-------------")
print(data[["height", "weight", "age"]].apply(np.log, axis=0))
print("-------------沿着1轴计算BMI指数-------------")


def BMI(series):
    weight = series["weight"]
    height = series["height"] / 100
    BMI = weight / height ** 2
    return BMI


print(data.apply(BMI, axis=1))
