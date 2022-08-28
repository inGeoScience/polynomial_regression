import numpy
import pandas
from matplotlib import pyplot
from sklearn import preprocessing


x = numpy.arange(0, 20, 1)
Y = numpy.cos(x/2)
Y = Y.reshape(-1, 1)
X = numpy.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
X = numpy.c_[numpy.ones(X.shape[0]), X]
theta = numpy.zeros((1, X.shape[1])) # theta为(1, 2)的二维数组。共有3个参数，包括截距项b在内
# 代价函数
def compCost(X, Y, theta):
    inner = numpy.power(numpy.matmul(X, theta.T) - Y, 2)
    return 1/(2*len(X))*numpy.sum(inner)
# Batch Gradient Descent Function
def batchGradientDescent(X, Y, theta, alpha, iters):
    # 创建临时参数二维数组
    temp_theta = numpy.zeros(theta.shape)
    # 创建代价列表(一维数组)以记录每一次代价，这里当作list来用
    cost = numpy.zeros(iters)
    for i in range(iters):
        error = numpy.matmul(X, theta.T) - Y
        # 计算每一个θ，并把θ添加到临时参数向量里。enumerate()返回一个tuple，包含两个元素——第i行和值。
        for nth_para, para in enumerate(theta.T):
            # 每次累加中乘的x^(i)是一个标量，所以用的numpy.multiply()，而不是*
            # X[:, nth_para]是一个一维数组，为了得到对应元素的点乘，把它转换成二维数组。
            para = para - ((alpha/len(X))*numpy.sum(numpy.multiply(error, X[:, nth_para].reshape(-1, 1))))
            temp_theta[0, nth_para] = para
        # 同步更新，并记录每一次代价
        theta = temp_theta
        cost[i] = compCost(X, Y, theta)
    return theta, cost
# 给定学习率、迭代次数
alpha = 0.1
iters = 1000000
# 进行梯度下降
theta, cost = batchGradientDescent(X, Y, theta, alpha, iters)
print(theta)
print(cost[-1])
# 出图
fig, ax = pyplot.subplots(1, 2)
ax[0].plot(numpy.arange(iters), cost, 'r')
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('cost')
ax[1].scatter(X[:, 1], Y)
ax[1].set_xlabel('x')
ax[1].plot(X[:, 1], numpy.matmul(X, theta.T), color='green')
# 为这两个图设置一个大标题
pyplot.suptitle('Multiple Linear Regression with Gradient Descent')
pyplot.show()