import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from sympy import symbols, diff
from math import log
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



# Data visualisation with 3D charts



# Minimize
# f(x, y) = 1/(3**(-x**2-y**2)+1)


def f(x, y):
    return 1/(3**(-x**2-y**2)+1)

def fpx(x, y):
    return 2*3**(-x**2 - y**2)*x*log(3)/(3**(-x**2 - y**2) + 1)**2

def fpy(x, y):
    return 2*3**(-x**2 - y**2)*y*log(3)/(3**(-x**2 - y**2) + 1)**2

# Make data
# x = np.linspace(start=-2, stop=2, num=200)
# y = np.linspace(start=-2, stop=2, num=200)
#
# x, y = np.meshgrid(x, y)
# Generating the 3D plot
# fig = plt.figure(figsize=[16, 12])
# ax = fig.gca(projection='3d')
#
# ax.set_xlabel("X", fontsize=20)
# ax.set_ylabel("Y", fontsize=20)
# ax.set_zlabel("f(x, y) - cost", fontsize=20)
# ax.plot_surface(x, y, f(x, y), cmap=cm.coolwarm, alpha=0.6)
#plt.show()


# Partial derivatives and symbolic computation
# a, b = symbols("x, y")
# print(f'Our cost function f(x, y) is: {f(a, b)}')
# print(f'Partial derivative wrt x is: {diff(f(a, b), a)}')
# print(f'Partial derivative wrt y is: {diff(f(a, b), b)}')
# print(f'Value of (f(x, y) at x=1.8 and y=1.0 is: {f(a, b).evalf(subs={a: 1.8, b: 1.0})}')
#
# print(f'Value of the slope wrt x at x=1.8 and y=1.0 is: {diff(f(a, b), a).evalf(subs={a: 1.8, b: 1.0})}')


# Batch gradient descent with sympy
# setup
# learning_rate = 0.1
# max_iter = 500
# params = np.array([1.8, 1.0])  # initial guess

# for n in range(max_iter):
#     gradient_x = fpx(params[0], params[1])
#     gradient_y = fpy(params[0], params[1])
#     gradients = np.array([gradient_x, gradient_y])
#     params = params - learning_rate * gradients
#
# # Results
# print(f'Values in gradient array {gradients}')
# print(f'Minimum occurs at x value of: {params[0]}')
# print(f'Minimum occurs at y value of: {params[1]}')
# print(f'The cost is {f(params[0], params[1])}')


# Graphing 3d gradient descent and adv. numpy arrays

# learning_rate = 0.1
# max_iter = 500
# params = np.array([1.8, 1.0])  # initial guess
# values_array = params.reshape(1, 2)
#
# for n in range(max_iter):
#     gradient_x = fpx(params[0], params[1])
#     gradient_y = fpy(params[0], params[1])
#     gradients = np.array([gradient_x, gradient_y])
#     params = params - learning_rate * gradients
#     values_array = np.append(arr=values_array, values=params.reshape(1, 2), axis=0)
#
# # Results
# print(f'Values in gradient array {gradients}')
# print(f'Minimum occurs at x value of: {params[0]}')
# print(f'Minimum occurs at y value of: {params[1]}')
# print(f'The cost is {f(params[0], params[1])}')
#
# fig = plt.figure(figsize=[16, 12])
# ax = fig.gca(projection='3d')
#
# ax.set_xlabel("X", fontsize=20)
# ax.set_ylabel("Y", fontsize=20)
# ax.set_zlabel("f(x, y) - cost", fontsize=20)
# ax.plot_surface(x, y, f(x, y), cmap=cm.coolwarm, alpha=0.6)
# ax.scatter(values_array[:, 0], values_array[:, 1], f(values_array[:, 0], values_array[:, 1]), color='black')
#plt.show()



# Advance Numpy array practice

# kirk = np.array([['captain', 'drums']])
# print(kirk.shape)
#
# hs_band = np.array([['Black space', 'mc'], ['quest love', 'guitar']])
#
# the_roots = np.append(arr=hs_band, values=kirk, axis=0)
# print(the_roots)
#
# the_roots = np.append(arr=the_roots, values=[['malik b', 'mc2']], axis=0)
# # Slicing arrays
# print(f'Printing nicknames {the_roots[:, 0]}')
# print(f'Print band roles: {the_roots[:, 1]}')




# Working with data and real cost function

# Mean squared error: a cost function for regression problems

# RSS = sum_{i=1}^n (y**i - h_theta * x**i)**2

# MSE = (sum_{i=1}^n (y**i - h_theta * x**i)**2) / 2

# Make sample data

x_5 = np.array([[0.1, 1.2, 2.4, 3.2, 4.1, 5.7, 6.5]]).transpose()
y_5 = np.array([1.7, 2.4, 3.5, 3.0, 6.1, 9.4, 8.2]).reshape(7, 1)

print(f'Shape of x_5: {x_5.shape}')
print(f'Shape of y_5: {y_5.shape}')

# Quick linear regression
regr = LinearRegression()
regr.fit(x_5, y_5)

print(f'Theta 0: {regr.intercept_[0]}')
print(f'Theta 1: {regr.coef_[0][0]}')

plt.scatter(x_5, y_5, s=50)
plt.plot(x_5, regr.predict(x_5), color='orange', linewidth=3)
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.show()

#y_hat = theta0 + theta1 * x
y_hat = regr.intercept_[0] + regr.coef_[0][0] * x_5
print(f'Estimated values y_hat are: {y_hat}')


def mse(y, y_hat):
    mse_calc = 1/len(y) * sum((y - y_hat)**2)
    return mse_calc

print(f'Manualy calculated MSE is: {mse(y_5, y_hat)}')
print(f'MSE regression using function is: {mean_squared_error(y_5, regr.predict(x_5))}')
