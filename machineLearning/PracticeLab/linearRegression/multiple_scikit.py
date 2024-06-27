# 练习使用scikit
import numpy as np

np.set_printoptions(precision=2)
from utils import *
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
# from lab_utils_multi import  load_house_data
import matplotlib.pyplot as plt

dlblue = '#0096ff';
dlorange = '#FF9300';
dldarkred = '#C00000';
dlmagenta = '#FF40FF';
dlpurple = '#7030A0';
plt.style.use('./deeplearning.mplstyle')


def linear_predict(x_train, y_train, x_test):
    # print y_train
    # print("Type of y_train:", type(y_train))
    # print("First five elements of y_train are:\n", y_train[:5])

    # print('The shape of x_train is:', x_train.shape)
    # print('The shape of y_train is: ', y_train.shape)
    # print('Number of training examples (m):', len(x_train))
    # print("Type of x_test:", type(x_test))
    # print("x_test:", x_test)
    linear_model = LinearRegression()
    # X must be a 2-D Matrix
    linear_model.fit(x_train, y_train)

    b = linear_model.intercept_
    w = linear_model.coef_
    print(f"w = {w:}, b = {b:0.2f}")
    # print(f"Prediction on training set:\n {linear_model.predict(x_train)[:5]}")
    # print(f"prediction using w,b:\n {(x_train @ w + b)[:5]}")
    # print(f"Target values \n {y_train[:5]}")
    # print("-------------------------")

    # print(x_test)
    # predict_model = linear_model.predict(x_test)
    # x_house_predict = predict_model[0]
    # print(f"type of predict_model : ${predict_model}")
    # print(f" predicted price of a house with ${x_test[0, 0]} sqft, ${x_test[0, 1]} bedrooms = ${x_house_predict:0.2f}")
    return linear_model.predict(x_train)


def show_data(x_train_data, y_train_data, predict_data):
    X_features = ['size(sqft)', 'bedrooms']
    fig, ax = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(x_train_data[:, i], y_train_data, label='target')
        ax[i].set_xlabel(X_features[i])
        ax[i].scatter(x_train_data[:, i], predict_data, color=dlorange, label='predict')
    ax[0].set_ylabel("Price");
    ax[0].legend();
    fig.suptitle("target versus prediction using z-score normalized model")
    plt.show()


x_train_data, y_train_data = load_data_multi()

x_test_data = np.array([[2534, 3], [1203, 3]])
print("------------------------------------------------------")
predict_data = linear_predict(x_train_data, y_train_data, x_test_data)
show_data(x_train_data, y_train_data, predict_data)

# 正规化 normalization
print("------------------------normalization------------------------------")
scaler = StandardScaler()
# 正规化 使用训练集fit并且转换训练集
x_norm = scaler.fit_transform(x_train_data)
print(scaler.mean_)
print(scaler.scale_)
# 正规化 测试集
x_test_data_norm = scaler.transform(x_test_data)
print("First five elements of x_norm are:\n", x_norm[:5])
predict_data_norm = linear_predict(x_norm, y_train_data, x_test_data_norm)
show_data(x_norm, y_train_data, predict_data_norm)
